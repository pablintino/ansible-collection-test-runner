import abc
import argparse
import logging
import os
import pathlib
import re
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
import typing
import venv

import yaml

logger = logging.getLogger()
console = logging.StreamHandler()
logger.addHandler(console)


class _LogPipe(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = False
        self.read_fd, self.write_fd = os.pipe()
        self.pipe_reader = os.fdopen(self.read_fd)
        self.start()
        self.output = ""

    def fileno(self) -> int:
        return self.write_fd

    def run(self):
        for line in iter(self.pipe_reader.readline, ""):
            self.output = self.output + line
            # Raw print, colors are in place for the console
            print(line.strip("\n"))

        self.pipe_reader.close()

    def close(self):
        os.close(self.write_fd)

    def __enter__(self):
        return self

    def __exit__(self, _, __, ___):
        self.close()


def _run_capture_command(
        command_list: typing.List[str], logger_instance, cwd=None, timeout=None, env=None
) -> typing.Tuple[int, str, bool]:
    working_dir = os.getcwd() if not cwd else cwd
    start_time = time.time()
    string_command = " ".join(command_list).strip()

    with _LogPipe() as pipe:
        logger_instance.info(
            "Running '%s on %s'",
            string_command,
            working_dir,
        )
        process = None
        try:
            process = subprocess.Popen(
                command_list,
                stdin=subprocess.DEVNULL,
                stdout=pipe,
                stderr=pipe,
                universal_newlines=True,
                cwd=working_dir,
                env=env,
            )
            process.wait(timeout=timeout)
            return process.returncode, pipe.output, False
        except subprocess.CalledProcessError as err:
            logger_instance.error(
                "Failed to execute %s. Exit code non-zero.", string_command
            )
            return err.returncode, pipe.output, False
        except OSError:
            logger_instance.error("Failed to execute %s", string_command)
            return 1, "", False
        except subprocess.TimeoutExpired:
            logger_instance.error(
                "Failed to execute %s. Timeout (%d)", string_command, timeout
            )
            return 1, pipe.output, True
        finally:
            if process:
                process.terminate()
            logger_instance.info(
                "Command '%s' took %f seconds to execute",
                string_command,
                (time.time() - start_time),
            )


def fetch_galaxy_yml(root_path):
    ansible_galaxy_file = pathlib.Path(root_path).joinpath("galaxy.yml")
    if not ansible_galaxy_file.is_file():
        raise SystemExit(f"Cannot locate the Ansible galaxy file {ansible_galaxy_file}")
    with open(ansible_galaxy_file, "r", encoding="utf-8") as galaxy_fd:
        return yaml.safe_load(galaxy_fd)


def install_collection_python_requirements(
        collection_base: pathlib.Path, python_bin: pathlib.Path = None
):
    bin_path = str(python_bin.absolute()) if python_bin else "python3"
    for path in collection_base.glob("*requirements.txt"):
        print(str(path.absolute()))
        subprocess.check_call(
            [bin_path, "-m", "pip", "install", "-r", str(path.absolute())],
            cwd=collection_base,
        )


class TestRunnerConfig:
    CONFIG_PROFILE_LOCAL = "local"
    CONFIG_PROFILES = [
        CONFIG_PROFILE_LOCAL,
    ]

    def __init__(self, file_path, run_args) -> None:
        with open(file_path, "r", encoding="utf-8") as runner_config_file:
            self.__config_dict = yaml.safe_load(runner_config_file)
        self.__run_args = run_args
        self.__validate()
        self.__parse()

    def __validate(self):
        if "profiles" not in self.__config_dict or not isinstance(
                self.__config_dict["profiles"], dict
        ):
            raise SystemExit("test-runner configuration contains no profiles dict")

        unrecognised_profile = next(
            (
                prof_name not in self.CONFIG_PROFILES
                for prof_name in self.__config_dict["profiles"].keys()
            ),
            None,
        )
        if unrecognised_profile:
            raise SystemExit(
                f"test-runner configuration contains an unrecognised profile {unrecognised_profile}"
            )

        run_profile = self.get_profile_name()
        if not run_profile or (
                run_profile not in self.__config_dict["profiles"].keys()
        ):
            raise SystemExit(
                "test-runner configuration doesn't support the selected profile"
            )

        # Check that if a scenario is given, we must ensure that
        # we only target a single role.
        # All and default are special cases that work on every role.
        if (
                self.get_target_scenarios() != ["all"] and
                self.get_target_scenarios() != ["default"]
        ) and len(self.get_target_roles()) != 1:
            raise SystemExit(
                "if a non default scenario is given a specific target role must be picked"
            )

    def __parse(self):
        self.profiles = self.__config_dict["profiles"]

    def get_profile_name(self):
        return self.__run_args.profile

    def get_profile_config(self):
        return self.profiles[self.get_profile_name()]

    def get_output_dir(self) -> pathlib.Path:
        return pathlib.Path(self.__run_args.output)

    def get_target_roles(self) -> typing.List[str]:
        return self.__run_args.roles or []

    def get_molecule_cmd(self) -> str:
        return self.__run_args.molecule_command

    def get_break_on_fail(self) -> bool:
        return self.__run_args.break_on_fail

    def get_isolated_env(self) -> bool:
        return self.__run_args.isolated_env

    def get_test_types(self) -> typing.List[str]:
        return self.__run_args.test_types

    def get_target_scenarios(self) -> typing.List[str]:
        return self.__run_args.scenarios

    def is_debug(self) -> bool:
        return self.__run_args.debug

    def get_global_timeout(self) -> int:
        return int(self.get_profile_config().get("test-timeout", 300))

    def get_molecule_timeout(self, role_name) -> int:
        profile = self.get_profile_config()
        role_timeout = profile.get("molecules-timeouts", {}).get(role_name, None)
        return int(
            role_timeout or profile.get("molecule-timeout", self.get_global_timeout())
        )


class BaseTestExecutor(metaclass=abc.ABCMeta):
    __REGEX_REMOVE_ANSI = r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"

    def __init__(
            self,
            runner_config: TestRunnerConfig,
            env_vars: typing.Dict[str, str],
            target_dir: pathlib.Path,
            roles: typing.List[str],
    ):
        self._runner_config = runner_config
        self._env_vars = env_vars
        self._target_dir = target_dir
        self._roles = roles
        self._logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def get_implemented_test_types(self):
        pass

    @abc.abstractmethod
    def run(self, test_type):
        pass

    def _write_to_output_file(self, file_name: str, content: str):
        output_path = self._runner_config.get_output_dir().joinpath(f"{file_name}.log")
        with open(output_path, "w", encoding="utf-8") as out_file:
            # Remove ANSI chars to get rid of color
            no_color_output = re.compile(self.__REGEX_REMOVE_ANSI).sub("", content)
            out_file.write(no_color_output)


class MoleculeTestExecutor(BaseTestExecutor):
    __TEST_TYPE_MOLECULE = "molecule"
    __BASE_TEST_TYPES = [__TEST_TYPE_MOLECULE]

    def __get_role_driver(self, role_name):
        profile_config = self._runner_config.get_profile_config()
        for driver, roles_list in profile_config.get("pinned-roles", {}).items():
            if role_name in roles_list:
                return driver

        profile = self._runner_config.get_profile_name()
        if "default-driver" not in profile_config:
            raise SystemExit(
                f"Profile {profile} has no default-driver and {role_name} is not pinned to a driver"
            )

        return profile_config["default-driver"]

    def __get_drivers_and_configs(self):
        roles_drivers = {
            role_name: self.__get_role_driver(role_name) for role_name in self._roles
        }
        drivers_config_paths = {}
        for driver in set(roles_drivers.values()):
            # If already added, skip
            if driver in drivers_config_paths:
                continue

            profile = self._runner_config.get_profile_name()
            driver_config_path = self._target_dir.joinpath(
                ".config", "molecule", f"config_{profile}_{driver}.yml"
            )
            if not driver_config_path.is_file():
                raise SystemExit(
                    f"Profile {profile} uses the {driver} driver "
                    "that has no configuration in .config/molecule dir"
                )
            drivers_config_paths[driver] = driver_config_path

        return roles_drivers, drivers_config_paths

    def _run_role_molecule(self, role_name, molecule_driver, driver_config) -> bool:
        succeed = True
        for scenario_name in self._runner_config.get_target_scenarios():
            succeed = (
                    self._run_role_scenario_molecule(driver_config, molecule_driver, role_name, scenario_name)
                    and succeed
            )
            if not succeed and self._runner_config.get_break_on_fail():
                return succeed
        return succeed

    def _run_role_scenario_molecule(self, driver_config, molecule_driver, role_name, scenario: str) -> bool:
        base_command = self._get_base_command(
            str(driver_config), self._runner_config.get_molecule_cmd(), scenario_name=scenario
        )

        role_path = self._target_dir.joinpath("roles", role_name)
        rc, output, timed_out = _run_capture_command(
            base_command,
            self._logger,
            cwd=role_path,
            timeout=self._runner_config.get_molecule_timeout(role_name),
            env=self._env_vars,
        )
        self._write_to_output_file(f"molecule_{molecule_driver}_{role_name}_{scenario}", output)
        if timed_out:
            # Try to clean up and ignore the output
            _run_capture_command(
                self._get_base_command(str(driver_config), "destroy"),
                self._logger,
                cwd=role_path,
                timeout=self._runner_config.get_molecule_timeout(role_name),
                env=self._env_vars,
            )
        return rc == 0

    def _get_base_command(self, config_path, command, scenario_name=None):
        cmd = [
            "molecule",
        ]
        if self._runner_config.is_debug():
            cmd.extend(["-vvv", "--debug"])
        cmd.extend(["-c", config_path, command])

        # These options go after the command argument
        if "all" == scenario_name:
            cmd.append("--all")
        elif scenario_name:
            cmd.append(f"--scenario-name={scenario_name}")

        return cmd

    def get_implemented_test_types(self):
        return self.__BASE_TEST_TYPES

    def run(self, test_type) -> bool:
        if test_type != self.__TEST_TYPE_MOLECULE:
            return True

        succeed = True
        roles_drivers, drivers_config_paths = self.__get_drivers_and_configs()
        for role in self._roles:
            molecule_driver = roles_drivers[role]
            driver_config = drivers_config_paths[molecule_driver]
            succeed = (
                    self._run_role_molecule(role, molecule_driver, driver_config)
                    and succeed
            )
            if not succeed and self._runner_config.get_break_on_fail():
                return succeed
        return succeed


class AnsibleTestExecutor(BaseTestExecutor):
    __BASE_TEST_TYPES = ["sanity", "units", "integration"]

    def get_implemented_test_types(self):
        return self.__BASE_TEST_TYPES

    def run(self, test_type):
        if test_type not in self.get_implemented_test_types():
            return True

        return self._run_ansible_test(test_type)

    def _run_ansible_test(self, test) -> bool:
        rc, output, _ = _run_capture_command(
            self._get_base_command(test),
            self._logger,
            cwd=self._target_dir,
            timeout=self._runner_config.get_global_timeout(),
            env=self._env_vars,
        )
        self._write_to_output_file(f"ansible_test_{test}.log", output)

        return rc == 0

    def _get_base_command(self, test_type):
        # Add the running python version as a target to avoid
        # running ansible-test on unwanted python versions (like 2.7)
        cmd = ["ansible-test", test_type, "--color=yes", "--requirements",
               f"--target-python={sys.version_info.major}.{sys.version_info.minor}"]
        if self._runner_config.is_debug():
            cmd.extend(["-vvv", "--debug"])
        return cmd


class TestRunner:
    __EXECUTORS_TABLE: typing.Dict[str, typing.List[type]] = {
        TestRunnerConfig.CONFIG_PROFILE_LOCAL: [
            AnsibleTestExecutor,
            MoleculeTestExecutor,
        ]
    }

    def __init__(self, runner_config: TestRunnerConfig, base_dir: pathlib.Path) -> None:
        self.__runner_config = runner_config
        self.__base_dir = base_dir
        self.__temp_collection_dir = None
        self.__roles = []

    def __build_executors_table(
            self, env_vars: typing.Dict[str, str]
    ) -> typing.Dict[str, BaseTestExecutor]:
        profile = self.__runner_config.get_profile_name()
        if profile not in self.__EXECUTORS_TABLE:
            raise SystemExit(f"{profile} profile not implemented")

        target_roles = self.__get_target_roles()
        executors = [
            executor(
                self.__runner_config,
                env_vars,
                self.__temp_collection_dir,
                target_roles,
            )
            for executor in self.__EXECUTORS_TABLE[profile]
        ]
        executors_table = {}
        for test_type in self.__runner_config.get_test_types():
            test_type_executor = next(
                (
                    executor
                    for executor in executors
                    if test_type in executor.get_implemented_test_types()
                ),
                None,
            )
            if not test_type_executor:
                raise SystemExit(f"Unsupported test type {test_type}")
            executors_table[test_type] = test_type_executor

        return executors_table

    def __discover_roles(self):
        roles_dir = self.__temp_collection_dir.joinpath("roles")
        if roles_dir.is_dir():
            for path in roles_dir.iterdir():
                if path.is_dir() and not path.name.startswith("."):
                    self.__roles.append(path.name)

    def __prepare_output_dir(self):
        logs_out_dir = self.__runner_config.get_output_dir()
        if not logs_out_dir.exists():
            logs_out_dir.mkdir()
        else:
            for path in logs_out_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()

    def __init_temp_dir(self, temporal_dir):
        galaxy_content = fetch_galaxy_yml(self.__base_dir)
        namespace = galaxy_content["namespace"]
        collection_name = galaxy_content["name"]
        self.__temp_collection_dir = pathlib.Path(temporal_dir).joinpath(
            "ansible_collections", namespace, collection_name
        )

        shutil.copytree(
            self.__base_dir,
            self.__temp_collection_dir,
            dirs_exist_ok=True,
            # Do not include .git as ignored, as ansible-test uses it
            # to compute the files that need to be scanned.
            # If a proper gitignore is missing, the fresh built
            # venv may be included.
            ignore=shutil.ignore_patterns(".venv", "venv"),
        )

    def __load_create_env(self) -> typing.Dict[str, str]:
        env_vars = dict(os.environ)
        if self.__runner_config.get_isolated_env():
            # Create the temporal python environment
            tmp_python_bin = self.__temp_collection_dir.joinpath(
                ".venv", "bin", "python3"
            )
            venv.create(self.__temp_collection_dir.joinpath(".venv"), with_pip=True)
            install_collection_python_requirements(
                self.__temp_collection_dir, tmp_python_bin
            )
            original_path = env_vars.get("PATH", "")
            env_vars[
                "PATH"
            ] = f"{tmp_python_bin.parent.absolute()}:{original_path}".strip(":")

        return env_vars

    def __get_target_roles(self) -> typing.List[str]:
        target_roles = self.__runner_config.get_target_roles()
        non_existing_role = next(
            (role for role in target_roles if role not in self.__roles), None
        )
        if non_existing_role:
            raise SystemExit(f"The selected role {non_existing_role} does not exist")

        return [
            role
            for role in self.__roles
            if ((not target_roles) or role in target_roles)
        ]

    def run(self):
        if self.__runner_config.is_debug():
            logger.setLevel("DEBUG")

        with tempfile.TemporaryDirectory() as temporal_dir:
            self.__init_temp_dir(temporal_dir)
            self.__prepare_output_dir()
            self.__discover_roles()

            succeed = True
            env_vars = self.__load_create_env()
            executors_table = self.__build_executors_table(env_vars)
            for test_type in self.__runner_config.get_test_types():
                succeed = executors_table[test_type].run(test_type) and succeed
                if not succeed and self.__runner_config.get_break_on_fail():
                    break
            raise SystemExit(0 if succeed else 1)


def __find_repo_root():
    try:
        return pathlib.Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
            ).strip()
        )
    except subprocess.CalledProcessError as err:
        raise SystemExit(
            "Cannot locate the repository root. Command "
            f"should be called inside a valid repo. {err}"
        ) from err


def __test_run(root_path, cli_args):
    runner_config_path = root_path.joinpath(".config", "testing.yml")
    if not runner_config_path.exists():
        raise SystemExit(
            f"Cannot locate the runner configuration in {runner_config_path.absolute()}"
        )

    runner_config = TestRunnerConfig(runner_config_path, cli_args)
    TestRunner(runner_config, root_path).run()


def __init_collection_symlink(root_path):
    base_venv_path = pathlib.Path(sys.prefix)
    site_pages = [
        pathlib.Path(path)
        for path in site.getsitepackages()
        if base_venv_path in pathlib.Path(path).parents
           and pathlib.Path(path).name == "site-packages"
    ]
    galaxy_content = fetch_galaxy_yml(root_path)
    for site_page_path in site_pages:
        namespace_dir = site_page_path.joinpath(
            "ansible_collections", galaxy_content["namespace"]
        )
        if not namespace_dir.exists():
            namespace_dir.mkdir(parents=True, exist_ok=True)

        if not namespace_dir.is_dir():
            raise SystemExit(
                f"The {namespace_dir} path in the venv must be a directory"
            )
        collection_dir = namespace_dir.joinpath(galaxy_content["name"])
        if not collection_dir.exists():
            collection_dir.symlink_to(root_path, target_is_directory=True)


def __init_run(root_path, _):
    install_collection_python_requirements(root_path)
    if sys.prefix != sys.base_prefix:
        # Running in a venv
        __init_collection_symlink(root_path)


def __init_run_command_argparse(subparsers):
    command_parser = subparsers.add_parser(
        "init",
        description=(
            "Setups the environment by adding a symlink fo the project root "
            "into the ansible_collections venv site-packages, allowing IDEs to work"
        ),
    )
    command_parser.set_defaults(func=__init_run, output=True, generate_vars=False)


def __test_run_command_argparse(subparsers, root_path):
    command_parser = subparsers.add_parser("test")
    command_parser.set_defaults(func=__test_run, output=True, generate_vars=False)

    command_parser.add_argument(
        "--profile",
        choices=TestRunnerConfig.CONFIG_PROFILES,
        help="Selected profile",
        required=True,
    )

    command_parser.add_argument(
        "--output",
        help="Testing output directory",
        default=str(root_path.joinpath("testing-output")),
    )

    command_parser.add_argument(
        "--roles",
        help="Run testing only for the given roles, comma separated",
        type=lambda t: [s.strip() for s in t.split(",")],
    )

    command_parser.add_argument(
        "--scenarios",
        help="The molecule scenarios to test",
        type=lambda t: [s.strip() for s in t.split(",")],
        default="all"
    )

    command_parser.add_argument(
        "--molecule-command",
        help="molecule test command to be run",
        default="test",
    )

    command_parser.add_argument(
        "--test-types",
        help="Run certain test types only, comma separated",
        default="sanity,units,integration,molecule",
        type=lambda t: [s.strip() for s in t.split(",")],
    )

    command_parser.add_argument(
        "--break-on-fail",
        help="Stop executing test phases as soon as one fails",
        default=True,
    )

    command_parser.add_argument(
        "--isolated-env",
        action="store_true",
        help="Use a fresh venv to run the tests",
    )
    command_parser.add_argument(
        "--no-isolated-env",
        dest="isolated_env",
        action="store_false",
        help="Don't use a fresh venv to run the tests",
    )
    command_parser.set_defaults(isolated_env=True)
    command_parser.add_argument("--debug", action="store_true")


def main():
    root_path = __find_repo_root()
    parser = argparse.ArgumentParser(
        prog="ansible test-runner",
        description="Manages the execution of tests of an Ansible repo",
        epilog="Text at the bottom of help",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    __test_run_command_argparse(subparsers, root_path)
    __init_run_command_argparse(subparsers)

    args = parser.parse_args()
    args.func(root_path, args)


main()

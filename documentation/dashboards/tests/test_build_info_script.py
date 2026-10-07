"""Tests for the build-identity generator (DrewsChessMachine/generate-build-info.sh).

Run: python3 -m unittest discover -s documentation/dashboards/tests

Each test copies the script into a temporary git repository laid out like this one
(`DrewsChessMachine/` holding the script, the build counter, the generated
`DrewsChessMachine/App/BuildInfo.swift` and a source file; `experiments/` beside it),
commits, runs the script with the toolchain build settings set, and parses the
generated Swift. The user's and system git configuration are kept out, so the
results depend only on the repository.
"""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SCRIPT = os.path.join(REPO, "DrewsChessMachine", "generate-build-info.sh")

COUNTER = "DrewsChessMachine/build_counter.txt"
BUILD_INFO = "DrewsChessMachine/DrewsChessMachine/App/BuildInfo.swift"
SOURCE = "DrewsChessMachine/DrewsChessMachine/Source.swift"
README = "experiments/README.md"

TOOLCHAIN = {
    "XCODE_PRODUCT_BUILD_VERSION": "17A5241e",
    "SDK_PRODUCT_BUILD_VERSION": "26A5300a",
    "CONFIGURATION": "Debug",
}


def git_environment():
    environment = dict(os.environ)
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    for name in TOOLCHAIN:
        environment.pop(name, None)
    return environment


class Repository:
    """A committed temporary repository with this repo's layout and the script under test."""

    def __init__(self, root):
        self.root = root
        self.write("DrewsChessMachine/generate-build-info.sh", open(SCRIPT, "rb").read())
        os.chmod(self.path("DrewsChessMachine/generate-build-info.sh"), 0o755)
        self.write(COUNTER, b"10\n")
        self.write(BUILD_INFO, b"// generated\n")
        self.write(SOURCE, b"let answer = 42\n")
        self.write(README, b"notes\n")
        self.git("init", "-q", "-b", "main")
        self.git("add", "-A")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com",
                 "commit", "-q", "-m", "initial")

    def path(self, relative):
        return os.path.join(self.root, relative)

    def write(self, relative, data):
        full = self.path(relative)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(data)

    def git(self, *arguments):
        return subprocess.run(["git", "-C", self.root, *arguments], check=True, capture_output=True,
                              env=git_environment()).stdout

    def run_script(self, toolchain=None):
        environment = git_environment()
        environment.update(TOOLCHAIN if toolchain is None else toolchain)
        return subprocess.run([self.path("DrewsChessMachine/generate-build-info.sh")], capture_output=True,
                              env=environment)

    def build_info(self, toolchain=None):
        result = self.run_script(toolchain)
        if result.returncode != 0:
            raise AssertionError(f"the script failed: {result.stderr.decode()}")
        return parse(open(self.path(BUILD_INFO)).read())


def parse(swift):
    def one(pattern):
        match = re.search(pattern, swift)
        if match is None:
            raise AssertionError(f"no match for {pattern!r} in:\n{swift}")
        return match.group(1)

    diff = one(r'static let gitDiffSHA256: String\? = (nil|"[0-9a-f]{64}")')
    return {
        "build_number": int(one(r"static let buildNumber = (\d+)")),
        "git_dirty": one(r"static let gitDirty = (true|false)") == "true",
        "git_diff_sha256": None if diff == "nil" else diff.strip('"'),
        "xcode_build": one(r'static let xcodeBuild = "([^"]*)"'),
        "sdk_build": one(r'static let sdkBuild = "([^"]*)"'),
        "configuration": one(r'static let configuration = "([^"]*)"'),
    }


def framed(path, content):
    encoded = path.encode()
    return b"%d\n%s\n%d\n%s" % (len(encoded), encoded, len(content), content)


class BuildInfoScriptTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.folder)

    def repository(self, name="repository"):
        root = os.path.join(self.folder, name)
        os.makedirs(root)
        return Repository(root)

    def test_a_clean_tree_is_not_dirty_after_the_counter_bump(self):
        repository = self.repository()
        info = repository.build_info()
        self.assertEqual(info["build_number"], 11)
        self.assertFalse(info["git_dirty"])
        self.assertIsNone(info["git_diff_sha256"])
        # A second build sees the counter and BuildInfo.swift the first one changed, and stays clean.
        again = repository.build_info()
        self.assertEqual(again["build_number"], 12)
        self.assertFalse(again["git_dirty"])
        self.assertIsNone(again["git_diff_sha256"])

    def test_a_change_outside_DrewsChessMachine_is_not_dirty(self):
        repository = self.repository()
        repository.write(README, b"edited notes\n")
        repository.write("documentation/new.md", b"an untracked document\n")
        info = repository.build_info()
        self.assertFalse(info["git_dirty"])
        self.assertIsNone(info["git_diff_sha256"])

    def test_a_tracked_source_edit_is_dirty_and_hashed(self):
        repository = self.repository()
        repository.write(SOURCE, b"let answer = 43\n")
        info = repository.build_info()
        self.assertTrue(info["git_dirty"])
        self.assertRegex(info["git_diff_sha256"], r"^[0-9a-f]{64}$")

    def test_a_staged_source_edit_is_dirty(self):
        repository = self.repository()
        repository.write(SOURCE, b"let answer = 44\n")
        repository.git("add", SOURCE)
        info = repository.build_info()
        self.assertTrue(info["git_dirty"])
        self.assertRegex(info["git_diff_sha256"], r"^[0-9a-f]{64}$")

    def test_an_untracked_source_file_is_dirty_and_changes_the_hash(self):
        repository = self.repository()
        repository.write(SOURCE, b"let answer = 43\n")
        edit_only = repository.build_info()["git_diff_sha256"]
        repository.write("DrewsChessMachine/DrewsChessMachine/New.swift", b"let added = 1\n")
        with_untracked = repository.build_info()
        self.assertTrue(with_untracked["git_dirty"])
        self.assertNotEqual(with_untracked["git_diff_sha256"], edit_only)

    def test_an_untracked_file_alone_hashes_its_framing(self):
        repository = self.repository()
        path = "DrewsChessMachine/DrewsChessMachine/New.swift"
        content = b"let added = 1\n"
        repository.write(path, content)
        info = repository.build_info()
        self.assertTrue(info["git_dirty"])
        self.assertEqual(info["git_diff_sha256"], hashlib.sha256(framed(path, content)).hexdigest())

    def test_untracked_files_are_framed_in_sorted_path_order(self):
        repository = self.repository()
        files = [("DrewsChessMachine/b.swift", b"b\n"), ("DrewsChessMachine/a.swift", b"a\n")]
        for path, content in files:
            repository.write(path, content)
        info = repository.build_info()
        expected = b"".join(framed(path, content) for path, content in sorted(files))
        self.assertEqual(info["git_diff_sha256"], hashlib.sha256(expected).hexdigest())

    def test_the_generated_files_never_change_the_hash(self):
        repository = self.repository()
        repository.write(SOURCE, b"let answer = 43\n")
        first = repository.build_info()
        # The run rewrote both generated files; edit them further by hand as well.
        repository.write(COUNTER, b"500\n")
        repository.write(BUILD_INFO, b"// something else entirely\n")
        second = repository.build_info()
        self.assertTrue(second["git_dirty"])
        self.assertEqual(second["git_diff_sha256"], first["git_diff_sha256"])

    def test_the_same_diff_hashes_the_same_twice(self):
        hashes = []
        for name in ("one", "two"):
            repository = self.repository(name)
            repository.write(SOURCE, b"let answer = 43\n")
            repository.write("DrewsChessMachine/DrewsChessMachine/New.swift", b"let added = 1\n")
            hashes.append(repository.build_info()["git_diff_sha256"])
        self.assertEqual(hashes[0], hashes[1])
        self.assertRegex(hashes[0], r"^[0-9a-f]{64}$")

    def test_framing_distinguishes_path_and_content_boundaries(self):
        # Path + content concatenate to the same bytes; the framing must still tell them apart.
        first = self.repository("first")
        first.write("DrewsChessMachine/u/a", b"bc")
        second = self.repository("second")
        second.write("DrewsChessMachine/u/ab", b"c")
        self.assertNotEqual(first.build_info()["git_diff_sha256"], second.build_info()["git_diff_sha256"])

    def test_toolchain_fields_are_written_and_empty_refuses(self):
        repository = self.repository()
        info = repository.build_info()
        self.assertEqual(info["xcode_build"], TOOLCHAIN["XCODE_PRODUCT_BUILD_VERSION"])
        self.assertEqual(info["sdk_build"], TOOLCHAIN["SDK_PRODUCT_BUILD_VERSION"])
        self.assertEqual(info["configuration"], TOOLCHAIN["CONFIGURATION"])
        for name in TOOLCHAIN:
            for value in (None, ""):
                toolchain = dict(TOOLCHAIN)
                if value is None:
                    del toolchain[name]
                else:
                    toolchain[name] = value
                before = open(repository.path(BUILD_INFO), "rb").read()
                result = repository.run_script(toolchain)
                self.assertNotEqual(result.returncode, 0, f"{name}={value!r} must fail the build")
                self.assertIn(name, result.stderr.decode())
                self.assertEqual(open(repository.path(BUILD_INFO), "rb").read(), before,
                                 f"{name}={value!r} must not write BuildInfo.swift")


if __name__ == "__main__":
    unittest.main()

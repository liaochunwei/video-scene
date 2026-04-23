use assert_cmd::Command;
use predicates::prelude::*;

fn video_split() -> Command {
    Command::cargo_bin("video-scene").unwrap()
}

#[test]
fn test_no_args_shows_help_hint() {
    video_split()
        .assert()
        .success()
        .stdout(predicate::str::contains("help"));
}

#[test]
fn test_help_flag() {
    video_split()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("video-scene"));
}

#[test]
fn test_workspace_init_and_list() {
    let tmp_dir = format!("/tmp/video-scene-test-ws-{}", std::process::id());
    let _ = std::fs::remove_dir_all(&tmp_dir);
    video_split()
        .arg("workspace")
        .arg("init")
        .arg("test-ws")
        .arg(&tmp_dir)
        .assert()
        .success();
    video_split()
        .arg("workspace")
        .arg("list")
        .assert()
        .success()
        .stdout(predicate::str::contains("test-ws"));
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_config_command() {
    video_split()
        .arg("config")
        .assert()
        .success();
}

#[test]
fn test_face_list() {
    video_split()
        .arg("face")
        .arg("list")
        .assert()
        .success();
}

#[test]
fn test_plugins_status_when_not_running() {
    let _ = std::process::Command::new("cargo")
        .args(["run", "--", "plugins", "exit"])
        .output();
    std::thread::sleep(std::time::Duration::from_secs(1));

    let output = std::process::Command::new("cargo")
        .args(["run", "--", "plugins", "status"])
        .output()
        .expect("Failed to run command");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("not running"), "Expected 'not running', got: {}", stdout);
}

#[test]
fn test_plugins_start_and_exit() {
    let _ = std::process::Command::new("cargo")
        .args(["run", "--", "plugins", "exit"])
        .output();
    std::thread::sleep(std::time::Duration::from_secs(1));

    let output = std::process::Command::new("cargo")
        .args(["run", "--", "plugins", "start"])
        .output()
        .expect("Failed to start daemon");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("started") || stdout.contains("already running"),
        "Expected 'started' or 'already running', got: {}", stdout);

    std::thread::sleep(std::time::Duration::from_secs(2));

    let output = std::process::Command::new("cargo")
        .args(["run", "--", "plugins", "status"])
        .output()
        .expect("Failed to check status");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("running"), "Expected 'running', got: {}", stdout);

    let output = std::process::Command::new("cargo")
        .args(["run", "--", "plugins", "exit"])
        .output()
        .expect("Failed to exit daemon");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("exiting") || stdout.contains("not running"),
        "Expected 'exiting' or 'not running', got: {}", stdout);
}

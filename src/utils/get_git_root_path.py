import subprocess
import os

def get_git_root():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Not in a git repository")


# Usage
if __name__ == "__main__":
    repo_root = get_git_root()
    print(repo_root)
import git
import os

def get_git_branch_gitpython():
    try:
        repo = git.Repo(os.getcwd())
        branch_name = repo.active_branch.name
        return branch_name
    except git.InvalidGitRepositoryError:
        return "Error: Not a valid Git repository."
    except Exception as e:
        return f"An error occurred: {e}"

current_branch = get_git_branch_gitpython()
print(f"You are currently using the '{current_branch}' branch.")
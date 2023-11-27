import subprocess
from os.path import expanduser
import time

HOME = expanduser("~")
# this is the commit hash of the odoo repo that we want to use during installation
ODOO_COMMIT_HASH = "0d99f4b9d9de404fea44ecf5480ec3b07c2b0fd6"
PATCH_PATH = "./convert.patch"
DB_NAME = "convert-attrs-and-states"
ODOO_HOME = HOME + "/odoo/dev/odoo"


addons_path = "addons/,../psbe-athem/"
addon_install = "base,athem" 

def git_stash_dir(path):
  """runs git stash on a directory and return false if there are no changes to stash"""
  print('git stash ' + path)
  return subprocess.check_output(['git', 'stash'], cwd=path).decode("utf-8").rstrip() != "No local changes to save"

def git_checkout_commit_hash(path, commit_hash):
  """runs git checkout on a directory"""
  print('git checkout ' + commit_hash + ' ' + path)
  subprocess.call(['git', 'checkout', commit_hash], cwd=path)


def git_apply_patch(path, patch_path):
  """runs git apply on a directory"""
  print('git apply ' + patch_path)
  subprocess.call(['git', 'apply', patch_path], cwd=path)


def reset_db(db_name):
  """resets a database by dropping it and recreating it"""
  print('running: dropdb ' + db_name)
  subprocess.call(['dropdb', db_name])
  print('running: createdb ' + db_name)
  subprocess.call(['createdb', db_name])

def run_odoo_install(path, addons_dir):
  """runs odoo install on an addons directory"""
  reset_db(DB_NAME)
  print("running: odoo-bin -d " + DB_NAME + " --addons-path " + addons_dir + " --stop-after-init")
  subprocess.call(['./odoo-bin', '-d', DB_NAME, '--addons-path', addons_dir, '-i base,athem', '--stop-after-init'], cwd=path)

def git_get_current_branch(path):
  """returns the current branch of a directory"""
  print('running: git rev-parse --abbrev-ref HEAD')
  return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=path).decode("utf-8").rstrip()

def git_checkout_branch(path, branch):
  """runs git checkout on a directory"""
  print('git checkout ' + branch)
  subprocess.call(['git', 'checkout', branch], cwd=path)

def git_stash_pop(path):
  """runs git stash pop on a directory"""
  print('git stash pop')
  subprocess.call(['git', 'stash', 'pop'], cwd=path)

def git_clear_changes(path):
  """runs git reset --hard on a directory"""
  print('git reset --hard')
  subprocess.call(['git', 'reset', '--hard'], cwd=path)

def main():
  print("fixing `attrs` and `states` fields in odoo")


  # # get the current branch
  current_branch = git_get_current_branch(ODOO_HOME)
  print('current branch: ' + current_branch)

  # stash the current branch
  changes_staged: bool = git_stash_dir(ODOO_HOME)
  # checkout the commit hash
  git_checkout_commit_hash(ODOO_HOME, ODOO_COMMIT_HASH)

  #  apply the patch
  git_apply_patch(ODOO_HOME, PATCH_PATH)

  # run odoo install
  run_odoo_install(ODOO_HOME, addons_path)

  # clear changes
  git_clear_changes(ODOO_HOME)

  #  checkout the current branch
  git_checkout_branch(ODOO_HOME, current_branch)

  # stash pop
  if changes_staged:
    git_stash_pop(path=ODOO_HOME)

main()



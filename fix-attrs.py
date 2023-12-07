import subprocess
from os.path import expanduser
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

HOME = expanduser("~")
DB_NAME = "convert-attrs-and-states"
DEFAULT_ODOO_HOME = HOME + "/odoo/dev/odoo"
REMOTE_BRANCH = "https://github.com/moaz-eldefrawy/odoo"
ENTERPRISE_COMMIT_HASH = "6f59d1cad6dac07e331e109181e083efaef78b68"

def git_stash_dir(path):
  """runs git stash on a directory and return false if there are no changes to stash"""
  logging.info('git stash ' + path)
  return subprocess.check_output(['git', 'stash'], cwd=path).decode("utf-8").rstrip() != "No local changes to save"

def git_checkout_commit_hash(path, commit_hash):
  """runs git checkout on a directory"""
  logging.info('git checkout ' + commit_hash + ' ' + path)
  subprocess.call(['git', 'checkout', commit_hash], cwd=path)


def git_apply_patch(path, patch_path):
  """runs git apply on a directory"""
  logging.info('git apply ' + patch_path)
  subprocess.call(['git', 'apply', patch_path], cwd=path)


def reset_db(db_name):
  """resets a database by dropping it and recreating it"""
  logging.info('running: dropdb ' + db_name)
  subprocess.call(['dropdb', db_name])
  logging.info('running: createdb ' + db_name)
  subprocess.call(['createdb', db_name])

def run_odoo_install(odoo_path, enterprise_path, addons_paths, addons_install):
  """runs odoo install on an addons directory"""
  odoo_addons_path = get_child_directory_path(odoo_path, "addons")
  enterprise_addons_path = enterprise_path
  full_addons_path = odoo_addons_path + "," + enterprise_addons_path + "," + addons_paths

  reset_db(DB_NAME)
  logging.info('running: odoo -i ' + addons_install + ' --addons-path=' + full_addons_path + ' -d ' + DB_NAME + "--stop-after-init")
  subprocess.call(['./odoo-bin', '-i', addons_install, '--addons-path=' + full_addons_path , '-d', DB_NAME, "--stop-after-init"], cwd=odoo_path)
  
def git_get_current_branch(path):
  """returns the current branch of a directory"""
  logging.info('running: getting current branch of ' + path + ' directory')
  return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=path).decode("utf-8").rstrip()

def git_checkout_branch(path, branch):
  """runs git checkout on a directory"""
  logging.info('git checkout ' + branch)
  subprocess.call(['git', 'checkout', branch], cwd=path)

def git_stash_pop(path):
  """runs git stash pop on a directory"""
  logging.info('git stash pop')
  subprocess.call(['git', 'stash', 'pop'], cwd=path)

def git_clear_changes(path):
  """runs git reset --hard on a directory"""
  logging.info('git reset --hard')
  subprocess.call(['git', 'reset', '--hard'], cwd=path)


def git_add_remote(path, remote_name, remote_url):
  """runs git remote add on a directory"""
  logging.info('git remote add ' + remote_name + ' ' + remote_url)
  subprocess.call(['git', 'remote', 'add', remote_name, remote_url], cwd=path)

def git_fetch_remote(path, remote_name):
  """runs git fetch on a directory"""
  logging.info('git fetch ' + remote_name)
  subprocess.call(['git', 'fetch', remote_name], cwd=path)

def get_current_dir():
  """returns the current directory"""
  return os.path.dirname(os.path.realpath(__file__))

def get_parent_directly(path):
  """returns the parent directory of a path"""
  return os.path.dirname(path)

def get_child_directory_path(path, child_dir_name):
  """returns the path of a child directory"""
  return os.path.join(path, child_dir_name)

def main():
  # declare variables
  odoo_path = None
  enterprise_path = None
  
  if len(sys.argv) > 1:
    addons_path = sys.argv[1]
  if len(sys.argv) > 2:
    addons_install = sys.argv[2]
  if len(sys.argv) > 3:
    odoo_path = sys.argv[3]
  if len(sys.argv) > 4:
    enterprise_path = sys.argv[4] 
  
  if not odoo_path:
    odoo_path = DEFAULT_ODOO_HOME

  if not enterprise_path:
    enterprise_path = get_child_directory_path(get_parent_directly(odoo_path), "enterprise")

  # check that file "odoo-bin" exists
  if not os.path.isfile(odoo_path + "/odoo-bin"):
    logging.error("odoo-bin file not found. Please provide the path to odoo directory")
    sys.exit(1)

  # check that file "enterprise" exists
  if not os.path.isdir(enterprise_path):
    logging.error("Enterprise directory not found. Provide the path for enterprise directory")
    sys.exit(1)
  
  # check that file "addons" exists
  if not addons_path and not addons_install:
    logging.error("addons_path or addons_install are not provided")
    sys.exit(1)

  logging.info("starting: fix `attrs` and `states` fields")

  odoo_current_branch = git_get_current_branch(odoo_path)
  enterprise_current_branch = git_get_current_branch(enterprise_path)
  logging.debug('odoo current branch: ' + odoo_current_branch)
  logging.info('enterprise current branch: ' + enterprise_current_branch)

  logging.info('stashing odoo and enterprise directories')
  odoo_changes_staged: bool = git_stash_dir(odoo_path)
  enterprise_changes_staged: bool = git_stash_dir(enterprise_path)

  # checkout the branch that has the fix
  git_add_remote(odoo_path, 'moaz-origin', REMOTE_BRANCH)
  git_fetch_remote(odoo_path, 'moaz-origin')

  # checkout the branches that has the fix
  git_checkout_branch(odoo_path, 'moaz-origin/17.0-fix_attrs_and_states-meld')
  git_checkout_branch(enterprise_path, ENTERPRISE_COMMIT_HASH)

  # run odoo to apply the conversion
  run_odoo_install(odoo_path, enterprise_path, addons_path, addons_install)

  # clear changes after curl
  git_clear_changes(odoo_path)

  # checkout the original branches
  git_checkout_branch(odoo_path, odoo_current_branch)
  git_checkout_branch(enterprise_path, enterprise_current_branch)

  logging.info('unstashing odoo and enterprise directories')
  if enterprise_changes_staged:
    git_stash_pop(path=enterprise_path)
  
  if odoo_changes_staged:
    git_stash_pop(path=odoo_path)


main()



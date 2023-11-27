import subprocess
from os.path import expanduser
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

HOME = expanduser("~")
DB_NAME = "convert-attrs-and-states"
ODOO_HOME = HOME + "/odoo/dev/odoo"
REMOTE_BRANCH = "https://github.com/moaz-eldefrawy/odoo"

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

def run_odoo_install(path, addons_paths, addons_install):
  """runs odoo install on an addons directory"""
  reset_db(DB_NAME)
  logging.info('running: odoo -i ' + addons_install + ' --addons-path=' + addons_paths + ' -d ' + DB_NAME)
  subprocess.call(['./odoo-bin', '-i', addons_install, '--addons-path=' + addons_paths, '-d', DB_NAME, "--stop-after-init"], cwd=path)
  
def git_get_current_branch(path):
  """returns the current branch of a directory"""
  logging.info('running: git rev-parse --abbrev-ref HEAD')
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


def main():
  if len(sys.argv) > 1:
    addons_path = sys.argv[1]
  if len(sys.argv) > 2:
    addons_install = sys.argv[2]

  # check that file "odoo-bin" exists
  if not os.path.isfile(ODOO_HOME + "/odoo-bin"):
    logging.error("odoo-bin file not found. Please run this in the odoo directory")
    sys.exit(1)
  
  if not addons_path and not addons_install:
    logging.error("addons_path or addons_install are not provided")
    sys.exit(1)

  logging.info("starting: fix `attrs` and `states` fields")

  current_branch = git_get_current_branch(ODOO_HOME)
  logging.info('current branch: ' + current_branch)

  changes_staged: bool = git_stash_dir(ODOO_HOME)

  # checkout the branch that has the fix
  git_add_remote(ODOO_HOME, 'moaz-origin', REMOTE_BRANCH)
  git_fetch_remote(ODOO_HOME, 'moaz-origin')
  git_checkout_branch(ODOO_HOME, 'moaz-origin/17.0-fix_attrs_and_states-meld')

  run_odoo_install(ODOO_HOME, addons_path, addons_install)

  git_clear_changes(ODOO_HOME)

  git_checkout_branch(ODOO_HOME, current_branch)

  if changes_staged:
    git_stash_pop(path=ODOO_HOME)


main()


# example: python3 fix-attrs.py addons/,../psbe-mintjens/ mintjens_sale_stock


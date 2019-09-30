# Some of this code adapted from axil/import-ipynb:
# https://github.com/axil/import-ipynb

import io, os, sys, types, tempfile
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


def register_colab_notebooks(id_map):
  """
  Register a set of Colab notebooks so they can be imported as Python modules.
  The notebooks must be saved to Google Drive.

  This function accepts a dictionary mapping module names to Google Drive IDs.
  For example:

  register_colab_notebooks({'foo': 'x7d19123', 'bar': '234dax9cx'})
  import foo
  import bar

  This code assumes that the Google Drive files with IDs x7d19123 and 234dax9cx
  are valid notebooks, and imports them as Python modules named foo and bar
  respectively.

  Inputs:
  - id_map: Dictionary mapping module names to Google Drive IDs
  """
  # Clear any existing ColabNotebookFinder objects from the path
  sys.meta_path = [x for x in sys.meta_path if not isinstance(x, ColabNotebookFinder)]
  sys.meta_path.append(ColabNotebookFinder(id_map))


class ColabLoader(object):
  def __init__(self, id_map):
    self.shell = InteractiveShell.instance()
    self.id_map = id_map 
  
  def load_module(self, name):
    drive_id = self.id_map.get(name, None)
    assert drive_id is not None
   
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
   
    fd, path = tempfile.mkstemp(suffix='.ipynb')
    os.close(fd)
   
    gfile = drive.CreateFile({'id': drive_id})
    gfile.GetContentFile(path)
   
    with io.open(path, 'r', encoding='utf-8') as f:
      nb = read(f, 4)
     
    mod = types.ModuleType(name)
    mod.__file__ = path
    mod.__loader__ = self
    mod.__dict__['get_ipython'] = get_ipython
    sys.modules[name] = mod
   
    save_user_ns = self.shell.user_ns
    self.shell.user_ns = mod.__dict__
   
    try:
      for cell in nb.cells:
        if cell.cell_type == 'code':
          itm = self.shell.input_transformer_manager
          code = itm.transform_cell(cell.source)
          # exec import, class, or function cell only
          if code.startswith('import ') or code.startswith('class ') or code.startswith('def ') or code.startswith('@'):
            exec(code, mod.__dict__)
    except Exception as e:
      print(e)
      print(code)
    finally:
      self.shell.user_ns = save_user_ns
    return mod
          

class ColabNotebookFinder(object):
  def __init__(self, id_map):
    self.id_map = id_map
  
  def find_module(self, fullname, path=None):
    if fullname not in self.id_map:
      return
   
    loader = ColabLoader(self.id_map)
    return loader


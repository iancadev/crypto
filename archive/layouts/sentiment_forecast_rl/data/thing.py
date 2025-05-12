#!/usr/bin/env python
# coding: utf-8

# In[10]:


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

display = lambda x: __import__('IPython').display.display(x) if is_notebook() else None


# In[11]:


display("HI")


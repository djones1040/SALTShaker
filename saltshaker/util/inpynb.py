try:
    cfg = get_ipython().config 
    in_ipynb=True

except NameError:
    in_ipynb = False

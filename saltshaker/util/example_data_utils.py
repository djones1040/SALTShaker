import os

def download_dir(remote_url, dirname):
    """
    Download a remote tar file to a local directory.

    Parameters
    ----------
    remote_url : str
        The URL of the file to download

    dirname : str
        Directory in which to place contents of tarfile. Created if it
        doesn't exist.

    Raises
    ------
    URLError (from urllib2 on PY2, urllib.request on PY3)
        Whenever there's a problem getting the remote file.
    """

    import io
    import tarfile

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    mode = 'r:gz' if remote_url.endswith(".tgz") else None

    # download file to buffer
    buf = io.BytesIO()
    _download_file(remote_url, buf)
    buf.seek(0)

    # create a tarfile with the buffer and extract
    tf = tarfile.open(fileobj=buf, mode=mode)
    tf.extractall(path=dirname)
    tf.close()
    buf.close()  # buf not closed when tf is closed.

def _download_file(remote_url, target):
    """
    Accepts a URL, downloads the file to a given open file object.

    This is a modified version of astropy.utils.data.download_file that
    downloads to an open file object instead of a cache directory.
    """

    from contextlib import closing
    from urllib.request import urlopen, Request
    from urllib.error import URLError, HTTPError
    from astropy.utils.console import ProgressBarOrSpinner


    timeout = 10.0
    download_block_size = 32768
    try:
        # Pretend to be a web browser (IE 6.0). Some servers that we download
        # from forbid access from programs.
        headers = {'User-Agent': 'Mozilla/5.0',
                   'Accept': ('text/html,application/xhtml+xml,'
                              'application/xml;q=0.9,*/*;q=0.8')}
        req = Request(remote_url, headers=headers)
        with closing(urlopen(req, timeout=timeout)) as remote:

            # get size of remote if available (for use in progress bar)
            info = remote.info()
            size = None
            if 'Content-Length' in info:
                try:
                    size = int(info['Content-Length'])
                except ValueError:
                    pass

            dlmsg = "Downloading {0}".format(remote_url)
            with ProgressBarOrSpinner(size, dlmsg) as p:
                bytes_read = 0
                block = remote.read(download_block_size)
                while block:
                    target.write(block)
                    bytes_read += len(block)
                    p.update(bytes_read)
                    block = remote.read(download_block_size)

    # Append a more informative error message to HTTPErrors, URLErrors.
    except HTTPError as e:
        e.msg = "{}. requested URL: {!r}".format(e.msg, remote_url)
        raise
    except URLError as e:
        append_msg = (hasattr(e, 'reason') and hasattr(e.reason, 'errno') and
                      e.reason.errno == 8)
        if append_msg:
            msg = "{0}. requested URL: {1}".format(e.reason.strerror,
                                                   remote_url)
            e.reason.strerror = msg
            e.reason.args = (e.reason.errno, msg)
        raise e

    # This isn't supposed to happen, but occasionally a socket.timeout gets
    # through.  It's supposed to be caught in `urrlib2` and raised in this
    # way, but for some reason in mysterious circumstances it doesn't. So
    # we'll just re-raise it here instead.
    except socket.timeout as e:
        # add the requested URL to the message (normally just 'timed out')
        e.args = ('requested URL {!r} timed out'.format(remote_url),)
        raise URLError(e)

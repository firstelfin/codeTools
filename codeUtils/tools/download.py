#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   download.py
@Time    :   2024/12/20 17:05:19
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

from urllib import request, parse

def is_url(url, check=False):
    """
    Validates if the given string is a URL and optionally checks if the URL exists online.
    Args:
        url (str): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.
            Defaults to False.
    Returns:
        (bool): Returns True for a valid URL. If 'check' is True, also returns True if the URL exists online.
            Returns False otherwise.
    Example:
        ```python
        valid = is_url("https://www.example.com")
        ```
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    except Exception:
        return False


# def safe_download(
#     url,
#     file=None,
#     dir=None,
#     curl=False,
#     retry=3,
#     min_bytes=1e0,
#     progress=True,
# ):
#     """
#     Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.
#     Args:
#         url (str): The URL of the file to be downloaded.
#         file (str, optional): The filename of the downloaded file.
#             If not provided, the file will be saved with the same name as the URL.
#         dir (str, optional): The directory to save the downloaded file.
#             If not provided, the file will be saved in the current working directory.
#         unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
#         delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
#         curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
#         retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
#         min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
#             a successful download. Default: 1E0.
#         exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
#         progress (bool, optional): Whether to display a progress bar during the download. Default: True.
#     Example:
#         ```python
#         from ultralytics.utils.downloads import safe_download
#         link = "https://ultralytics.com/assets/bus.jpg"
#         path = safe_download(link)
#         ```
#     """
#     gdrive = url.startswith("https://drive.google.com/")  # check if the URL is a Google Drive link
#     if gdrive:
#         url, file = get_google_drive_file_info(url)
#     f = Path(dir or ".") / (file or url2file(url))  # URL converted to filename
#     if "://" not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
#         f = Path(url)  # filename
#     elif not f.is_file():  # URL and file do not exist
#         uri = (url if gdrive else clean_url(url)).replace(  # cleaned and aliased url
#             "https://github.com/ultralytics/assets/releases/download/v0.0.0/",
#             "https://ultralytics.com/assets/",  # assets alias
#         )
#         desc = f"Downloading {uri} to '{f}'"
#         LOGGER.info(f"{desc}...")
#         f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
#         check_disk_space(url, path=f.parent)
#         for i in range(retry + 1):
#             try:
#                 if curl or i > 0:  # curl download with retry, continue
#                     s = "sS" * (not progress)  # silent
#                     r = subprocess.run(["curl", "-#", f"-{s}L", url, "-o", f, "--retry", "3", "-C", "-"]).returncode
#                     assert r == 0, f"Curl return value {r}"
#                 else:  # urllib download
#                     method = "torch"
#                     if method == "torch":
#                         torch.hub.download_url_to_file(url, f, progress=progress)
#                     else:
#                         with request.urlopen(url) as response, TQDM(
#                             total=int(response.getheader("Content-Length", 0)),
#                             desc=desc,
#                             disable=not progress,
#                             unit="B",
#                             unit_scale=True,
#                             unit_divisor=1024,
#                         ) as pbar:
#                             with open(f, "wb") as f_opened:
#                                 for data in response:
#                                     f_opened.write(data)
#                                     pbar.update(len(data))
#                 if f.exists():
#                     if f.stat().st_size > min_bytes:
#                         break  # success
#                     f.unlink()  # remove partial downloads
#             except Exception as e:
#                 if i == 0 and not is_online():
#                     raise ConnectionError(emojis(f"❌  Download failure for {uri}. Environment is not online.")) from e
#                 elif i >= retry:
#                     raise ConnectionError(emojis(f"❌  Download failure for {uri}. Retry limit reached.")) from e
#                 LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {uri}...")

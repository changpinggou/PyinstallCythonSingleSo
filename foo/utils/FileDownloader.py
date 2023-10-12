import os
import shutil
import requests



class FileDownloader:
    def __init__(self, file_url, save_dir, filename=None):
        self.file_url = file_url
        self.save_dir = save_dir
        self.filename = filename or self._get_filename_from_url(self.file_url)
        self.filepath = os.path.join(self.save_dir, self.filename)

    def _get_filename_from_url(self, url):
        return os.path.basename(url)

    def download(self):
        # 兼容一下, 如果 self.file_url 是本地路径(以/开头,且在本地存在), 那就直接把 self.file_url 复制到 self.filepath
        if self.file_url.startswith("/") and os.path.exists(self.file_url):
            shutil.copy(self.file_url, self.filepath)
            return self.filepath
        # 走网络下载
        return self.download_from_url()

    def download_from_url(self):
        try:
            # Check if the file already exists
            if os.path.exists(self.filepath):
                raise ValueError(f"File already exists at {self.filepath}")

            response = requests.get(self.file_url, stream=True)
            # 判断文件是否存在
            if response.status_code == 404:
                raise ValueError(f"File not exists at {self.file_url}")
            
            with open(self.filepath, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        out_file.write(chunk)

            print(f"Successfully downloaded file from {self.file_url} to {self.filepath}")
            return self.filepath

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error downloading file: {str(e)}")


# # Usage example
# url = "https://example.com/file.zip"
# save_to = "/path/to/save/directory"
# filename = "my_file.zip"
# downloader = FileDownloader(url, save_to, filename)
# try:
#     file_path = downloader.download()
#     print(f"File downloaded successfully: {file_path}")
# except ValueError as e:
#     print(str(e))

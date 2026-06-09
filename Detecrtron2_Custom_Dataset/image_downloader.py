
from simple_image_download import simple_image_download as sp


response = sp.simple_image_download()


keywords = "underground coal miners"  
limit = 50  


response().download(keywords=keywords, limit=limit)
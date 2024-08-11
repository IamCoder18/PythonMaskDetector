from simple_image_download import simple_image_download as simp
response = simp.simple_image_download
lst=['real person image']
for rep in lst:
    print()
    response().download(rep, 50)
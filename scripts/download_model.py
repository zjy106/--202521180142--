from modelscope import snapshot_download
model_dir = snapshot_download('google/gemma-3-270m-it', cache_dir='./models')
print(f"模型已下载至: {model_dir}")
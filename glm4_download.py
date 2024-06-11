from modelscope import snapshot_download

if __name__ == '__main__':
    osmodel_dir = snapshot_download('ZhipuAI/glm-4-9b-chat', cache_dir='./', revision='master')

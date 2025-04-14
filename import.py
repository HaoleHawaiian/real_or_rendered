import pip

## Define install package
def install(package):
    pip.main(['install', kagglehub])

## Try to import kagglehub. If it's not installed, install it.
try:
    import kagglehub
except ImportError:
    print 'kagglehub is not installed. Installing it now.'
    install('kagglehub')


## Import kagglehub
import kagglehub

# Download latest version of the data
path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")

print("Path to dataset files:", path)

#!/usr/bin/python

try:
	import scipy
except ImportError as IE:
	print str(IE)
else:
	print('scipy: {0:>16}'.format(scipy.__version__))

try:
	import numpy
except ImportError as IE:
	print str(IE)
else:
	print('numpy: {0:>16}'.format(numpy.__version__))

try:
	import matplotlib
except ImportError as IE:
	print str(IE)
else:
	print('matplotlib: {0:>10}'.format(matplotlib.__version__))

# try:
# 	import pandas
# except ImportError as IE:
# 	print str(IE)
# else:
# 	print('pandas: {0:>15}'.format(pandas.__version__))

try:
	import sklearn
except ImportError as IE:
	print str(IE)
else:
	print('sklearn: {0:>14}'.format(sklearn.__version__))

try:
	import skimage
except ImportError as IE:
	print str(IE)
else:
	print('skimage: {0:>14}'.format(skimage.__version__))

try:
	import theano
except ImportError as IE:
	print str(IE)
else:
	print ('theano: {0:>59}'.format(theano.__version__))

try:
	import tensorflow
except ImportError as IE:
	print str(IE)
else:
	print ('tensorflow: {0:>11}'.format(tensorflow.__version__))

try:
	import keras
except ImportError as IE:
	print str(IE)
else:
	print ('keras: {0:>15}'.format(keras.__version__))
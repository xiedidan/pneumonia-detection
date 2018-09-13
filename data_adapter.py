import caffe

class PythonDataAdapter(caffe.Layer):
    def setup(self, bottom, top):
        assert len(top) == 2

        self.topNames=['data', 'label']

        # read parameters from `self.param_str`
        params = eval(self.param_str)
        self.params = params

        # reshape images
        top[0].reshape(params['batch_size'], 1, params['size'][0], params['size'][1])

    def reshape(self, bottom, top):
        # only reshape gts online
        top[1].reshape(self.params['batch_size'], -1, 5)

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

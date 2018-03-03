class common(object):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def tfidf_m(self):
        return self._tfidf_m

    @tfidf_m.setter
    def tfidf_m(self, value):
        self._tfidf_m = value

    @property
    def tfidf_d(self):
        return self._tfidf_d

    @tfidf_d.setter
    def tfidf_d(self, value):
        self._tfidf_d = value

    @property
    def tf_d(self):
        return self._tf_d

    @tf_d.setter
    def tf_d(self, value):
        self._tf_d = value

    @property
    def tf_m(self):
        return self._tf_m

    @tf_m.setter
    def tf_m(self, value):
        self._tf_m = value

    @property
    def lr_m(self):
        return self._lr_m

    @lr_m.setter
    def lr_m(self, value):
        self._lr_m = value


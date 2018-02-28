class common(object):

    def __init__(self, tfidf_m, lr_m):
        self.__tfidf_m = tfidf_m
        self.__lr_m = lr_m

    @property
    def tfidf_m(self):
        return self.__tfidf_m

    @property
    def lr_m(self):
        return self.__lr_m

import cfg
from compute_topics import get_kmeans, show_cluster_topics, get_lda, show_topics
from plot_data import plot_scatter_2d, get_svd, get_tsne


class analyze_data(object):

    def __init__(self, common):
        self._common = common

    def analyze_kmeans(self):
        # Compute topics using Kmeans
        kmean_m, kmean_d = get_kmeans(self._common.tfidf_d, cfg.n_topics, scale=False)

        # Show cluster top 15 words per topic
        print("Top 15 stemmed words per cluster in Kmeans model\n")
        show_cluster_topics(kmean_d, self._common.tfidf_d, self._common.tfidf_m.get_feature_names(), 15)

        # Prepare data for plotting
        svd_v, svd_m = get_svd(self._common.tfidf_d, 50)
        tnse_v, tsne_m = get_tsne(svd_m, 2, 25)

        # Plot Data
        # matplotlib inline
        plot_scatter_2d(tsne_m[0], tsne_m[1], kmean_d, 1000,
                        'KMeans Clustering of Amazon Reviews using TFIDF (t-SNE Plot)')

    def analyze_lda(self):
        # Compute topics using LDA
        lda_m, lda_d = get_lda(self._common.tf_d, cfg.n_topics)

        # Show cluster top 15 words per topic
        print("Top 15 stemmed words per topic in LDA model\n")
        show_topics(lda_m, self._common.tf_m.get_feature_names(), 15)

        # Prepare data for plotting
        svd_v, svd_m = get_svd(self._common.tfidf_d, 50)
        tnse_v, tsne_m = get_tsne(svd_m, 2, 25)
        lda_c = lda_d.argmax(axis=1)

        # matplotlib inline
        plot_scatter_2d(tsne_m[0], tsne_m[1], lda_c, 1000, 'LDA Topics of Amazon Reviews using TF (t-SNE Plot)')

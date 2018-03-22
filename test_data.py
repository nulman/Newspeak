import cfg


def test_review(common, text):
    test_str = [text]
    test_new = common.tfidf_m.transform(test_str)

    print('\nReview text: "{R}"\n'.format(R=test_str[0]))
    print('Model Predction')
    for m in range(0, cfg.categories.__len__()):
        print('Model ({M}): {P:.1%}'.format(M=cfg.categories[m], P=common.lr_m[m].predict_proba(test_new)[0][1]))

import re
import math


def getwords(doc):

    words = [s.lower() for s in doc.split()
           if len(s)>2 and len(s)<20]

    return dict([(w,1) for w in words])

class classifier:
    def __init__(self, getfeatures, filename=None):
        self.fc = {}
        self.cc = {}
        self.getfeatures = getfeatures

    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 0)
        self.fc[f][cat] += 1

    def incc(self, cat):
        self.cc.setdefault(cat, 0)
        self.cc[cat] += 1

    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return float(self.fc[f][cat])
        return 0.0

    def catcount(self, cat):
        if cat in self.cc:
            return float(self.cc[cat])
        return 0

    def totalcount(self):
        return sum(self.cc.values())

    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item)
        for f in features:
            self.incf(f, cat)

        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0
        return self.fcount(f, cat) / self.catcount(cat)

    def weightprob(self, f, cat, prf, weight=1, ap=0.5):
        basicprob = prf(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        bp = (weight * ap + totals * basicprob) / (weight + totals)
        return bp


class naivebayes(classifier):

    def __init__(self, getfeatures):
        super().__init__(getfeatures)
        self.thresholds = {}

    def setthreshold(self, cat, t):
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    def docprob(self, item, cat):
        features = self.getfeatures(item)
        p = 1
        for f in features:
            p *= self.weightprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return catprob * docprob

    def classify(self, item, default=None):
        probs = {}
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.getthreshold(best) > probs[best]:
                return default
            return best


class fisherclassifier(classifier):


    def __init__(self, getfeatures):
        super().__init__(getfeatures)
        self.minimums = {}

    def cprob(self, f, cat):
        clf = self.fprob(f, cat)
        if clf == 0 :
            return 0

        freqsum = sum([self.fprob(f, c) for c in self.categories()])

        p = clf / freqsum

        return p

    def fisherprob(self, item, cat):

        p = 1
        features = self.getfeatures(item)
        for f in features:
            p *= self.weightprob(f, cat, self.cprob)

        fscore = -2*math.log(p)
        return self.invchi2(fscore, len(features) * 2)

    def invchi2(self, chi, df):
        m = chi / 2.0
        sum = term = math.exp(-m)
        for i in range(1, df // 2):
            term *= m / i
            sum += term
        return min(sum, 1.0)

    def setminimum(self, cat, min):
        self.minimums[cat] = min

    def getminimum(self, cat):
        if cat not in self.minimums:
            return 0
        return self.minimums[cat]

    def classify(self, item, default=None):
        max = 0.0
        best = default
        for c in self.categories():
            p = self.fisherprob(item, c)
            if p > self.getminimum(c) and p > max:
                max = p
                best = c
        return best



def sampletrain(cl):
    cl.train('Nobody owns the water.', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy pharmaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick brown fox jumps', 'good')

c1 = naivebayes(getwords)

# sampletrain(c1)
# print(c1.classify("quick rabbit", default="unknown"))
# print(c1.classify("quick money", default="unknown"))
# c1.setthreshold("bad", 3.0)
# print(c1.classify("quick money", default="unknown"))
# for i in range(10):
#     sampletrain(c1)
# print(c1.classify("quick money", default="unknown"))

c2 = fisherclassifier(getwords)

sampletrain(c2)

print(c2.classify('quick rabbit'))
print(c2.classify('quick money'))
c2.setminimum('bad', 0.8)
print(c2.classify('quick money'))
c2.setminimum('good', 0.4)
print(c2.classify('quick money'))
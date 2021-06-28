class Rule(object):
    od = int
    associate_items = []

    def __init__(self, od, associate_items):
        self.od = od
        self.associate_items = associate_items

    def __repr__(self):
        return "od: %s - associate_items: %s" % (self.od, self.associate_items)


def new_rule(od, associate_items):
    rule = Rule(od, associate_items)
    return rule

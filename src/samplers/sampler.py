class Sampler:
    def __init__(self, gen_size, load_size, gen_sample_foo, load_sample_foo):
        self.gen_size = gen_size
        self.load_size = load_size
        self.gen_sample_foo = gen_sample_foo
        self.load_sample_foo = load_sample_foo

    def sample(self):
        while True:
            if self.gen_size > 0:
                gen_sample = self.gen_sample_foo()
            else:
                gen_sample = None
            if self.load_size > 0:
                load_sample = self.load_sample_foo()
            else:
                load_sample = None
            yield self.format_sample((gen_sample, load_sample))

    def get_sub_sample_len(self, sample):
        try:
            if type(sample) == type({}):
                subsample_len = sum([len(sample[k]) for k in sample.keys()])
            else:
                subsample_len = len(sample)
        except TypeError:
            subsample_len = 0
        return subsample_len

    def get_sample_len(self, sample):
        return self.get_sub_sample_len(sample[0]) + self.get_sub_sample_len(sample[1])
        

    def format_sample(self, sample):
        # dividing sample into gen and load parts
        if type(sample) == tuple:
            assert len(sample) == 2, "only gen & load separation is supported"
            assert (self.gen_size + 2 * self.load_size) == self.get_sample_len(
                sample
            ), "sample size does not match defined gen subsample and load subsample sizes"
            format_dict = {"Gen": sample[0], "Load": sample[1]}
        return format_dict

class Sampler:
    """Processing class to properly encapsulate and sampling process"""

    def __init__(self, gen_size, load_size, gen_sample_foo, load_sample_foo):
        """Initializer of an instance of this class. Includes information about number of generator plants and loads
            that have uncertain power injection contribution

        Args:
            gen_size (int): number of uncertain generation units
            load_size (int): number of loads that has uncertainty
            gen_sample_foo (function): returns a sample of uncertain generation
            load_sample_foo (function): returns a sample of uncertain load
        """
        self.gen_size = gen_size
        self.load_size = load_size
        self.gen_sample_foo = gen_sample_foo
        self.load_sample_foo = load_sample_foo

    def sample(self):
        """Sample generator

        Yields:
            Dict: dictionary that is formatted according to `self.format_sample` method
        """
        while True:
            # sample if uncertainties on generator plants or loads are specified
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
        """Returns sample size of generator or load. Util function to process sample format -- see `self.format_sample`

        Args:
            sample (Dict or np.array): presented sample

        Returns:
            int: sub sample size (of generators or loads)
        """
        try:
            if type(sample) == type(
                {}
            ):  # for loads, since it is a Dict {"P": ..., "Q": ...}
                subsample_len = sum([len(sample[k]) for k in sample.keys()])
            else:  # for generators which is a list
                subsample_len = len(sample)
        except TypeError:
            subsample_len = 0
        return subsample_len

    def get_sample_len(self, sample):
        """Returns total sample length

        Args:
            sample (Dict): formatted sample -- see `self.format_sample`

        Returns:
            int: Total sample length
        """
        return self.get_sub_sample_len(sample[0]) + self.get_sub_sample_len(sample[1])

    def format_sample(self, sample):
        """Formats sample into dictionary
        {"Gen": ...,
        "Load": {"P": ..., "Q": ...}
        }

        Args:
            sample (tuple): first element - sample for generators, seconds - sample for loads

        Returns:
            [type]: [description]
        """
        # dividing sample into gen and load parts
        if type(sample) == tuple:
            assert len(sample) == 2, "only gen & load separation is supported"
            assert (self.gen_size + 2 * self.load_size) == self.get_sample_len(
                sample
            ), "sample size does not match defined gen subsample and load subsample sizes"
            format_dict = {"Gen": sample[0], "Load": sample[1]}
        return format_dict

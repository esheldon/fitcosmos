import meds
import ngmix
import ngmix.medsreaders

def process_fofs(args):
    processor = Processor(args)
    processor.go()


class Processor(object):
    def __init__(self, args):
        self.args=args
        self._load_meds_files()

    def go(self):
        pass

    def _load_meds_files(self):
        mlist=[]
        for f in self.args.meds:
            print('loading:',f)
            mlist.append( meds.MEDS(f) )

        self.mb_meds = ngmix.medsreaders.MultiBandNGMixMEDS(
            mlist,
        )

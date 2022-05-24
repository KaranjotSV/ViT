from model import create_ViT, run

modes = [('F', 'F', 'F', 'F'),
         ('F', 'T', 'F', 'T'),
         ('F', 'T', 'T', 'F'),
         ('F', 'T', 'T', 'T'),
         ('T', 'F', 'F', 'F'),
         ('T', 'T', 'T', 'T')]


def to_flag(flag):
    if flag == 'T':
        return True
    elif flag == 'F':
        return False
    else:
        return 'invalid'


for mode in modes:
    flags = [to_flag(flag) for flag in mode]

    model = create_ViT(SPT=flags[0], LSA=flags[1], MASKING=flags[2], TRAIN_TAU=flags[3])
    history = run(model)

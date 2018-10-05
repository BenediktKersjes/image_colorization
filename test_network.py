import os
import matplotlib.pyplot as plt
import glob

from config import trained_models_path
from amlproject import TrainNetwork


class TestNetwork(TrainNetwork):
    def __init__(self, **kwargs):
        kwargs['do_not_log'] = True
        super(TestNetwork, self).__init__(**kwargs)

    def test(self):
        test_data, test_target = self.get_next_test_batch()
        test_output = self.model(test_data)
        for image in test_output:
            plt.imshow((image.max(dim=0)[1]).detach().cpu().numpy())
            plt.show()

        print(test_output.shape)
        for t in [.000001, .77, .58, .38, .29, .14, .01, .001]:
            images = self.convert_to_images(test_data, test_output, False, t=t)
            plt.imshow(images[0])
            plt.title(t)
            plt.show()


if __name__ == '__main__':
    _, test_file = os.path.split(max(glob.iglob(trained_models_path + '*'), key=os.path.getctime))
    print('load file {}'.format(test_file))

    tester = TestNetwork(
        batch_size=30,
        image_size=128,
        load_model=test_file,
        iterations_start=0,
        lr=1e-4,
        regression=False,
        loss='MultinomialCrossEntropyLoss',
        network='ColorfulImageColorization',
        convert_on_gpu=True)
    tester.test()

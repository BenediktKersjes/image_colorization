import random


def create_image_sequence(start, stop, algorithm):
    count = 0
    file = open('./data/shuffled.txt')

    for line in file:
        index = int(line.split(' ')[0])
        file = (line.split(' ')[1]).strip()

        if index < start:
            continue

        if index >= stop:
            break

        if random.choice([0, 1]) == 0:
            gt_side = 'left'
            right_image = algorithm + '/' + file
            left_image = 'gt/' + file
        else:
            gt_side = 'right'
            left_image = algorithm + '/' + file
            right_image = 'gt/' + file

        print('sequence_helper("{}","{}","{}");'.format(gt_side, left_image, right_image))
        count += 1

    if 472 > start and 472 < stop:
        file = 'beach00002494.jpg'

        if random.choice([0, 1]) == 0:
            gt_side = 'left'
            right_image = algorithm + '/' + file
            left_image = 'gt/' + file
        else:
            gt_side = 'right'
            left_image = algorithm + '/' + file
            right_image = 'gt/' + file

            print('sequence_helper("{}","{}","{}");'.format(gt_side, left_image, right_image))
        count += 1

    return count

def create_input_sequence(count):
    for i in range(count):
        print('<input type="hidden" name="selection{}" id="selection{}" value="unset">'.format((i+1), (i+1)))
        print('<input type="hidden" name="gt{}" id="gt{}" value="unset">'.format((i + 1), (i + 1)))
        print('<input type="hidden" name="correct{}" id="correct{}" value="unset">'.format((i + 1), (i + 1)))


if __name__ == '__main__':
    start = 1100
    create_input_sequence(20 * 4 + 4)

    print('')
    print('')
    print('')

    count = 0
    count += create_image_sequence(0, 1, 'colorful_mse_small')
    count += create_image_sequence(1, 2, 'colorful_classification_small')
    count += create_image_sequence(2, 3, 'koala_mse_small')
    count += create_image_sequence(3, 4, 'koala_classification_small')

    count += create_image_sequence(start, start+20, 'colorful_mse_small')
    count += create_image_sequence(start+20, start+40, 'colorful_classification_small')
    count += create_image_sequence(start+40, start+60, 'koala_mse_small')
    count += create_image_sequence(start+60, start+80, 'koala_classification_small')

    print(count)
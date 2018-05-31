from data_providers import QuickDrawImageDataProvider
import numpy as np

rng = np.random.RandomState(seed=0)  # set seed
batch_size = 100
train_data = QuickDrawImageDataProvider(which_set="train", batch_size=batch_size, rng=rng, num_classes_use=100)
val_data = QuickDrawImageDataProvider(which_set="valid", batch_size=batch_size, rng=rng, num_classes_use=100)
test_data = QuickDrawImageDataProvider(which_set="test", batch_size=batch_size, rng=rng, num_classes_use=100)




from data_providers import QuickDrawStrokeDataProvider
import numpy as np

rng = np.random.RandomState(seed=0)  # set seed
batch_size = 100
val_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)



from data_providers import QuickDrawCombinedDataProvider
import numpy as np

rng = np.random.RandomState(seed=0)  # set seed
batch_size = 100
val_data = QuickDrawCombinedDataProvider(which_set="valid", batch_size=batch_size, rng=rng)

from data_providers import QuickDrawCombinedDataProvider
import numpy as np

rng = np.random.RandomState(seed=0)  # set seed
batch_size = 100
train_data = QuickDrawCombinedDataProvider(which_set="train", batch_size=batch_size, rng=rng)






train_data = QuickDrawStrokeDataProvider(which_set="train", batch_size=batch_size, rng=rng)
val_data = QuickDrawStrokeDataProvider(whival_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)
val_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)
val_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)
val_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)
ch_set="valid", batch_size=batch_size, rng=rng)
test_data = QuickDrawStrokeDataProvider(which_set="test", batch_size=batch_size, rng=rng)

from data_providers import QuickDrawStrokeDataProvider
import numpy as np

rng = np.random.RandomState(seed=0)  # set seed
batch_size = 100
val_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)


train_data = QuickDrawStrokeDataProvider(which_set="train", batch_size=batch_size, rng=rng)
val_data = QuickDrawStrokeDataProvider(which_set="valid", batch_size=batch_size, rng=rng)
test_data = QuickDrawStrokeDataProvider(which_set="test", batch_size=batch_size, rng=rng)

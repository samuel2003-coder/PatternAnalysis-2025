train_loader = get_loader("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train")
val_loader   = get_loader("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate", shuffle=False)
test_loader  = get_loader("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test", shuffle=False)


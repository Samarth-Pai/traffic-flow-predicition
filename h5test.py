import h5py

# Open the file in read mode
with h5py.File('METR-LA.h5', 'r') as file:
    # View the groups inside the file
    print("Groups:", list(file.keys()))
    print(list(file.items())[0])
    for i in list(file.items())[0]:
        print(*i)
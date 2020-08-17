import matplotlib.pyplot as plt

fh = open("data.log", "r")
lines = fh.readlines()
fh.close()

numloops = []
cpus = []
gpus = []
gpus_2d = []
gpus_pl = []
gpus_pl_2d = []
gpus_um = []
for line in lines:
    buffer = line.split()
    numloops.append(int(buffer[0]))
    cpus.append(float(buffer[1]))
    gpus.append(float(buffer[2]))
    gpus_2d.append(float(buffer[3]))
    gpus_pl.append(float(buffer[4]))
    gpus_pl_2d.append(float(buffer[5]))
    gpus_um.append(float(buffer[6]))


fig, ax = plt.subplots(facecolor="b")
plt.xlabel("Number of loops")
plt.ylabel("Processing time [s]")
ax.set_xscale('log')
# ax.set_xlim(1, 10000)
# ax.set_ylim(0, 700)

plt.plot(numloops, cpus, color="black", label="CPU", marker="o", linestyle="-")
plt.plot(numloops, gpus, color="coral", label="GPU", marker="o", linestyle="-")
plt.plot(numloops, gpus_2d, color="lightcoral", label="GPU ( 2D )", marker="o", linestyle="-")
plt.plot(numloops, gpus_pl, color="springgreen", label="GPU by page-lock", marker="o", linestyle="-")
plt.plot(numloops, gpus_pl_2d, color="turquoise", label="GPU by page-lock ( 2D )", marker="o", linestyle="-")
plt.plot(numloops, gpus_um, color="dodgerblue", label="GPU on unified memory", marker="o", linestyle="-")

plt.legend()
# plt.show()
plt.savefig("time.png")
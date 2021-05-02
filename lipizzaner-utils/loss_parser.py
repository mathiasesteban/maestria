import re
import sys
import matplotlib.pyplot as plt

"""
Loss plotter, search for f(Generator(x)) and f(Discriminator(x)) in Lipizzaner loss LOGS.

$ python <path_to_file>

"""


g_loss = []
d_loss = []

# Open file
with open(sys.argv[1], 'r') as f:
    # Read each line
    lines = f.readlines()



counter = 0
# Iterate each line
for line in lines:

    # Find loss log line
    match = re.search('Iteration=\d*, f', line)


    if match:

        counter += 1
        # Find values
        generator_loss = float(re.search('f\(Generator\(x\)\)=(\d+\..+?),', line).group(1))
        discriminator_loss = float(re.search('f\(Discriminator\(x\)\)=(\d+\..+?),', line).group(1))

        # Parser exception
        if generator_loss is None or discriminator_loss is None:
            raise Exception("Parser error")

        print("Discriminator loss: %s - Generator loss: %s" % (discriminator_loss, generator_loss))

        g_loss.append(generator_loss)
        d_loss.append(discriminator_loss)

print("************************************")
print("Total lines: " + str(len(lines)))
print("Loss lines found: " + str(counter))
print("************************************")

# # Plot data

font_size = 20

plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size)  # legend fontsize

x = range(len(d_loss)) if len(d_loss) > 0 else range(len(g_loss))
plt.plot(x, d_loss, color='orange', label='Discriminator loss')
plt.plot(x, g_loss, color='#5488A5', label='Generator loss')
plt.legend(loc='lower right')


filename = re.search("(.*).log",sys.argv[1]).group(1)

plt.savefig("imgs/" + filename + "_loss" + '.png')
plt.savefig("imgs/" + filename + "_loss" + '.pdf')

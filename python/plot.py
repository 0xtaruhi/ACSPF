import vecfitpy
from matplotlib import pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO
)

network = vecfitpy.Network(filepath="./examples/1848-191031.s384p")
logging.info(f"Network loaded, shape: {network.sparams.shape}")

model = vecfitpy.Model(filepath="./model_384.dat")
logging.info("Model loaded")

res = model.eval(network)
print(res)

freqs = network.freqs
fitted = model.cal_response(freqs)

logging.info("Calulated model's response")

port1 = 123
port2 = 194

sparams00 = network.sparams[:, port1, port2]
fitted00 = fitted[:, port1, port2]

plt.plot(network.freqs, abs(sparams00))
plt.plot(network.freqs, abs(fitted00))

plt.savefig("sprams00.png", dpi=300)
"""Standalone plot: 20 waypoints in 3D, colored by how many of the 5 repeated
attempts (run_experimento_repeticiones) succeeded at each one.

Uso: python result_3d.py
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the 3D projector
import numpy as np

# =====================================================================
# 1. DATA -- waypoints (mm) and successes out of 5 attempts, read off the
# printed RESUMEN (order matches WP 1..20).
# =====================================================================
WAYPOINTS_MM = np.array([
    [-189.4, 221.2, 82.2],
    [160.9, 244.6, -83.1],
    [68.6, 233.4, 151.2],
    [-74.2, 237, -171.4],
    [-57, 277.3, 1.9],
    [40.2, 278.7, -17.3],
    [-0.9, 270.9, 71.6],
    [-16, 282.3, -23.8],
    [105.4, 288.9, -38.7],
    [-80, 247, 23.1],
    [-36, 275.2, -137.7],
    [93.6, 280.7, 76.1],
    [77.7, 303.3, 51.2],
    [-14.7, 304.4, -80.8],
    [53.9, 294.6, 6],
    [-112, 264.1, 100.6],
    [-105.1, 257.1, -120],
    [73.4, 269.4, -115.8],
    [-109.2, 248.5, -52.2],
    [-50.8, 269.5, 96.9],
])

# Successes out of 5 attempts per waypoint, tallied from the RESUMEN printout.
ACIERTOS = np.array([5, 5, 4, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 5, 5, 2, 5, 5, 0, 5])

assert len(WAYPOINTS_MM) == len(ACIERTOS) == 20

# =====================================================================
# 2. BUCKETS: set USE_4_TIERS to switch between the two schemes discussed.
# =====================================================================
USE_4_TIERS = True  # False -> simple 3-tier scheme (red/orange/green)


def bucket_4_tiers(aciertos):
  if aciertos == 0:
    return '0/5 (fail)', '#dc2626', 'X'
  if aciertos <= 2:
    return '1-2/5', '#f97316', 's'
  if aciertos <= 4:
    return '3-4/5', '#eab308', '^'
  return '5/5 (all succeeded)', '#16a34a', 'o'


def bucket_3_tiers(aciertos):
  if aciertos == 0:
    return '0/5 (fail)', '#dc2626', 'X'
  if aciertos <= 4:
    return '1-4/5', '#f97316', 's'
  return '5/5 (all succeeded)', '#16a34a', 'o'


bucket_fn = bucket_4_tiers if USE_4_TIERS else bucket_3_tiers

# =====================================================================
# 3. PLOT
# =====================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Group points by bucket so each category gets ONE legend entry (not 20).
categorias = {}
for i, (punto, aciertos) in enumerate(zip(WAYPOINTS_MM, ACIERTOS)):
  etiqueta, color, marker = bucket_fn(int(aciertos))
  categorias.setdefault(etiqueta, {'color': color, 'marker': marker, 'puntos': [], 'idx': []})
  categorias[etiqueta]['puntos'].append(punto)
  categorias[etiqueta]['idx'].append(i)

# Fixed legend order: worst to best.
orden_leyenda = (
    ['0/5 (fail)', '1-2/5', '3-4/5', '5/5 (all succeeded)']
    if USE_4_TIERS else
    ['0/5 (fail)', '1-4/5', '5/5 (all succeeded)']
)

for etiqueta in orden_leyenda:
  if etiqueta not in categorias:
    continue
  info = categorias[etiqueta]
  pts = np.array(info['puntos'])
  ax.scatter(
      pts[:, 0], pts[:, 1], pts[:, 2],
      color=info['color'], marker=info['marker'], s=110,
      edgecolors='black', linewidths=0.6,
      label=f"{etiqueta} (n={len(pts)})",
  )

# Label each point with its waypoint number.
for i, punto in enumerate(WAYPOINTS_MM):
  ax.text(punto[0], punto[1], punto[2], f'  {i + 1}', fontsize=8)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Waypoint success rate over 5 repeated attempts')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()

out_path = 'result_3d.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved figure to: {out_path}')

plt.show()

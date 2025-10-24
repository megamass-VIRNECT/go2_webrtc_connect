"""
최근 맵들 빠르게 확인
"""
import pickle
import numpy as np
import glob

# 최근 맵 파일들 찾기
map_files = sorted(glob.glob('go2_map*.pkl'), key=lambda x: -int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)

print("최근 맵 파일들 분석:")
print("=" * 80)

for map_file in map_files[:10]:  # 최근 10개만
    try:
        with open(map_file, 'rb') as f:
            grid = pickle.load(f)

        data = grid.data
        unknown = np.sum(data == -1)
        free = np.sum((data >= 0) & (data < 50))
        occupied = np.sum(data >= 50)
        total = data.size

        print(f"\n{map_file}:")
        print(f"  해상도: {grid.resolution}m, 크기: {grid.width}x{grid.height}")
        print(f"  Unknown: {unknown} ({unknown/total*100:.1f}%)")
        print(f"  Free: {free} ({free/total*100:.1f}%)")
        print(f"  Occupied: {occupied} ({occupied/total*100:.1f}%)")

        if occupied > 0:
            occupied_values = data[data >= 50]
            print(f"  Occupied 값 범위: [{occupied_values.min()}, {occupied_values.max()}], 평균: {occupied_values.mean():.1f}")

    except Exception as e:
        print(f"\n{map_file}: 오류 - {e}")

print("\n" + "=" * 80)

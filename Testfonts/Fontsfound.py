import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置字体文件路径
font_path = "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"

# 创建 FontProperties 对象
font_prop = fm.FontProperties(fname=font_path)


# 示例图表
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('测试图表 - Noto Sans CJK', fontproperties=font_prop)
plt.show()

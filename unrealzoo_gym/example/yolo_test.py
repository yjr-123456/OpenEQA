# from ultralytics import YOLOWorld
# import cv2
# import numpy as np

# yolomodel = YOLOWorld('yolov8x-worldv2.pt')
# yolomodel.set_classes(["white_drone", "person"])

# def predict(image_path):
#     results = yolomodel.predict(image_path, conf=0.25, iou=0.45, max_det=100)
#     return results

# def minimal_labels(image_path):
#     """最小化标签显示 - 淡化版本"""
#     results = yolomodel.predict(image_path, conf=0.25, iou=0.45, max_det=100)
    
#     img = cv2.imread(image_path)
#     annotated_img = img.copy()
    
#     if results[0].boxes is not None:
#         # 创建透明覆盖层
#         overlay = annotated_img.copy()
        
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#             confidence = float(box.conf.item())
#             cls_id = int(box.cls.item())
#             class_name = results[0].names[cls_id]
            
#             # 不同颜色表示不同类别
#             colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
#             color = colors[cls_id % len(colors)]
            
#             # ✅ 绘制较细的边框到覆盖层
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)  # 线宽从2改为1
            
#             # ✅ 绘制更小的圆圈 - 半径从15改为8
#             cv2.circle(overlay, (x1, y1), 8, color, -1)
            
#             # ✅ 使用更小的字体
#             cv2.putText(overlay, str(i+1), (x1-3, y1+3), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)  # 字体大小从0.5改为0.3，粗细从2改为1
        
#         # ✅ 应用透明度 - 让标签变淡
#         alpha = 0.7  # 透明度，0.0=完全透明，1.0=完全不透明
#         annotated_img = cv2.addWeighted(overlay, alpha, annotated_img, 1-alpha, 0)
        
#         # ✅ 优化图例显示 - 也让它变淡
#         legend_y = img.shape[0] - 25  # 稍微上移
#         legend_overlay = annotated_img.copy()
        
#         # 更小的背景条
#         cv2.rectangle(legend_overlay, (0, legend_y-15), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        
#         legend_text = ""
#         for i, box in enumerate(results[0].boxes):
#             cls_id = int(box.cls.item())
#             confidence = float(box.conf.item())
#             class_name = results[0].names[cls_id]
#             legend_text += f"{i+1}:{class_name}({confidence:.2f}) "
        
#         # ✅ 更小的图例字体
#         cv2.putText(legend_overlay, legend_text, (10, legend_y-2), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)  # 字体从0.4改为0.35
        
#         # ✅ 图例也应用透明度
#         legend_alpha = 0.8
#         annotated_img = cv2.addWeighted(legend_overlay, legend_alpha, annotated_img, 1-legend_alpha, 0)
        
#     else:
#         # 如果没有检测到任何对象，添加淡化的提示
#         overlay = annotated_img.copy()
#         cv2.putText(overlay, "No objects detected", 
#                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         annotated_img = cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0)
    
#     return results, annotated_img

# def ultra_minimal_labels(image_path):
#     """超级简化版本 - 只有边框和小点"""
#     results = yolomodel.predict(image_path, conf=0.25, iou=0.45, max_det=100)
    
#     img = cv2.imread(image_path)
#     annotated_img = img.copy()
    
#     if results[0].boxes is not None:
#         overlay = annotated_img.copy()
        
#         for i, box in enumerate(results[0].boxes):
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#             cls_id = int(box.cls.item())
            
#             colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]  # 使用更柔和的颜色
#             color = colors[cls_id % len(colors)]
            
#             # ✅ 只绘制很细的边框
#             cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            
#             # ✅ 只在角落放一个很小的点
#             cv2.circle(overlay, (x1+5, y1+5), 3, color, -1)  # 超小圆点
        
#         # ✅ 应用高透明度
#         alpha = 0.5  # 更高的透明度
#         annotated_img = cv2.addWeighted(overlay, alpha, annotated_img, 1-alpha, 0)
    
#     return results, annotated_img

# if __name__ == "__main__":
#     image_path = 'E:/EQA/unrealzoo_gym/example/QA_Data/SuburbNeighborhood_Day/counting/G3TCSP/obs_G3TCSP/obs_2.png'
    
#     print("=== 淡化版本 ===")
#     results1, img1 = minimal_labels(image_path)
    
#     print("=== 超级简化版本 ===")
#     results2, img2 = ultra_minimal_labels(image_path)
    
#     # 显示对比
#     cv2.imshow("Faded Minimal Labels", img1)
#     cv2.imshow("Ultra Minimal Labels", img2)
    
#     # 也可以看原始版本对比
#     original_annotated = results1[0].plot()
#     cv2.imshow("Original YOLO Plot", original_annotated)
    
#     # 打印检测结果
#     if results1[0].boxes is not None:
#         print(f"检测到 {len(results1[0].boxes)} 个对象:")
#         for i, box in enumerate(results1[0].boxes):
#             cls_id = int(box.cls.item())
#             confidence = float(box.conf.item())
#             class_name = results1[0].names[cls_id]
#             bbox = box.xyxy[0].cpu().numpy()
#             print(f"  {i+1}. {class_name}: 置信度={confidence:.3f}, 位置=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
#     else:
#         print("未检测到任何对象")
    
#     print("按任意键关闭窗口...")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

print([[0,0]]*8)
import psycopg2
import json
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")



class DBManager:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn=dsn)
        self.cursor = self.conn.cursor()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def rollback(self):
        self.conn.rollback()

db_manager = DBManager(DATABASE_URL)

def insert_image(path):
    query = "INSERT INTO image (path) VALUES (%s) ON CONFLICT (path) DO NOTHING;"
    db_manager.cursor.execute(query, (path,))
    db_manager.commit()

def get_image_id(path):
    query = "SELECT image_id FROM image WHERE path = %s;"
    db_manager.cursor.execute(query, (path,))
    result = db_manager.cursor.fetchone()
    return result[0] if result else None

def  insert_pose(pose_data):
    image_id = pose_data["image_id"]
    pose_json = pose_data["pose_json"]
    bbox_top_x = pose_data["bbox_top_x"]
    bbox_top_y = pose_data["bbox_top_y"]
    bbox_bottom_x = pose_data["bbox_bottom_x"]
    bbox_bottom_y = pose_data["bbox_bottom_y"]
    person_num = pose_data["person_num"]
    norm_joint= pose_data["norm_joint"]


    query = """
    INSERT INTO poses_2d (image_id, pose_json, bbox_top_x, bbox_top_y, bbox_bottom_x, bbox_bottom_y, person_num, norm_joint)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    on conflict (image_id, person_num)
    do update set pose_json = EXCLUDED.pose_json,
               bbox_top_x = EXCLUDED.bbox_top_x,
                bbox_top_y = EXCLUDED.bbox_top_y,   
                bbox_bottom_x = EXCLUDED.bbox_bottom_x,
                bbox_bottom_y = EXCLUDED.bbox_bottom_y,
                norm_joint = EXCLUDED.norm_joint
    ;
    """
    db_manager.cursor.execute(query, (image_id, pose_json, bbox_top_x, bbox_top_y, bbox_bottom_x, bbox_bottom_y, person_num, norm_joint))
    db_manager.commit()





def insert_pose_vector(pose_vector_data):
    image_id = pose_vector_data["image_id"]
    pose_vector = pose_vector_data["pose_vector"]
    person_num = pose_vector_data["person_num"]

    query = """
    INSERT INTO poses_2d (image_id, pose_vec, person_num)
    VALUES (%s, %s, %s)
    on conflict (image_id, person_num)
    do update set pose_vec = EXCLUDED.pose_vec;
    """
    db_manager.cursor.execute(query, (image_id, pose_vector.tolist(), person_num))
    db_manager.commit()




def get_result(vec, limit=20, offset=0):
    db_manager.cursor.execute("SET hnsw.ef_search = 200;")
    db_manager.cursor.execute("SET hnsw.iterative_scan = strict_order;")

    query = """
    SELECT
        p.poses_id,
        p.pose_vec <=> (%s)::vector AS dist,
        p.bbox_top_x, p.bbox_top_y, p.bbox_bottom_x, p.bbox_bottom_y
        FROM poses p, image i
        WHERE p.image_id = i.image_id
        ORDER BY dist
        LIMIT %s OFFSET %s;
    """
    veclist = vec.tolist()
    vecstr = "[" + ",".join(map(str, veclist)) + "]"
    db_manager.cursor.execute(query, (vecstr, limit, offset))
    
    results = db_manager.cursor.fetchall()
    topk = []

    for row in results:
        pose_id,_, bbox_top_x, bbox_top_y, bbox_bottom_x, bbox_bottom_y = row

        topk.append({
            "pose_id": str(pose_id),
            "bbox_top_x": bbox_top_x,
            "bbox_top_y": bbox_top_y,
            "bbox_bottom_x": bbox_bottom_x,
            "bbox_bottom_y": bbox_bottom_y,
        })
    return topk

def get_all_image_bbox():
    query = """
    SELECT
        p.poses_id,
        i.path,
        p.bbox_top_x, p.bbox_top_y, p.bbox_bottom_x, p.bbox_bottom_y
        FROM poses_2d p, image i
        WHERE p.image_id = i.image_id;
    """
    db_manager.cursor.execute(query)
    
    results = db_manager.cursor.fetchall()

    return results


def get_all_2d_poses():
    query = """
    SELECT
        p.poses_id,
        p.pose_json
        FROM poses_2d p;
    """
    db_manager.cursor.execute(query)
    
    results = db_manager.cursor.fetchall()

    return results

def insert_pose_3d_vector(pose_2d_id, pose_vector):
    query = """
    INSERT INTO poses_3d (pose_2d_id, pose_vec)
    VALUES (%s, %s)
    on conflict (pose_2d_id)
    do update set pose_vec = EXCLUDED.pose_vec;
    """
    db_manager.cursor.execute(query, (pose_2d_id, pose_vector.tolist()))
    db_manager.commit()



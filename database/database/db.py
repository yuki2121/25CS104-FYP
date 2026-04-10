import psycopg2
import json
import os
from dotenv import load_dotenv
from .storage_signer import sign_image_url
from psycopg2.pool import ThreadedConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

db_pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=5,
    dsn=DATABASE_URL,
)

def insert_image(path):
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            query = "INSERT INTO image (path) VALUES (%s) ON CONFLICT (path) DO NOTHING;"
            cursor.execute(query, (path,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)

def get_image_id(path):
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            query = "SELECT image_id FROM image WHERE path = %s;"
            cursor.execute(query, (path,))
            result = cursor.fetchone()
            return result[0] if result else None
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)

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
    INSERT INTO poses (image_id, pose_json, bbox_top_x, bbox_top_y, bbox_bottom_x, bbox_bottom_y, person_num, norm_joint)
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
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (image_id, pose_json, bbox_top_x, bbox_top_y, bbox_bottom_x, bbox_bottom_y, person_num, norm_joint))
        conn.commit()
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)





def insert_pose_vector(pose_vector_data):
    image_id = pose_vector_data["image_id"]
    pose_vector = pose_vector_data["pose_vector"]
    person_num = pose_vector_data["person_num"]

    query = """
    INSERT INTO poses (image_id, pose_vec, person_num)
    VALUES (%s, %s, %s)
    on conflict (image_id, person_num)
    do update set pose_vec = EXCLUDED.pose_vec;
    """
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (image_id, pose_vector.tolist(), person_num))
        conn.commit()
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)




def get_result(vec, limit=20, offset=0):
    conn = db_pool.getconn()
    close_conn = False
    try:
        veclist = vec.tolist()
        vecstr = "[" + ",".join(map(str, veclist)) + "]"

        query = """
        SELECT
            p.poses_id,
            p.pose_vec <=> (%s)::vector AS dist,
            p.bbox_top_x, p.bbox_top_y, p.bbox_bottom_x, p.bbox_bottom_y
        FROM poses_2d p
        JOIN image i ON p.image_id = i.image_id
        ORDER BY dist
        LIMIT %s OFFSET %s;
        """

        with conn.cursor() as cursor:
            cursor.execute("SET hnsw.ef_search = 200;")
            cursor.execute("SET hnsw.iterative_scan = strict_order;")
            cursor.execute(query, (vecstr, limit, offset))
            results = cursor.fetchall()

        topk = []
        for row in results:
            pose_id, _, bbox_top_x, bbox_top_y, bbox_bottom_x, bbox_bottom_y = row
            object_name = f"thumbs/{pose_id}.jpg"
            topk.append({
                "pose_id": str(pose_id),
                "url": f"/api/image/{object_name}",
                "bbox_top_x": bbox_top_x,
                "bbox_top_y": bbox_top_y,
                "bbox_bottom_x": bbox_bottom_x,
                "bbox_bottom_y": bbox_bottom_y,
            })
        return topk

    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)
        
        
        
def get_all_image_bbox():
    query = """
    SELECT
        p.poses_id,
        i.path,
        p.bbox_top_x, p.bbox_top_y, p.bbox_bottom_x, p.bbox_bottom_y
        FROM poses p, image i
        WHERE p.image_id = i.image_id;
    """
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        return results
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)


def get_all_2d_poses():
    query = """
    SELECT
        p.poses_id,
        p.pose_json
        FROM poses p;
    """
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        return results
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)
    


def insert_pose_3d_vector(pose_2d_id, pose_vector):
    query = """
    INSERT INTO poses_3d (pose_2d_id, pose_vec)
    VALUES (%s, %s)
    on conflict (pose_2d_id)
    do update set pose_vec = EXCLUDED.pose_vec;
    """
    conn = db_pool.getconn()
    close_conn = False
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (pose_2d_id, pose_vector.tolist()))
        conn.commit()
    except Exception as e:
        conn.rollback()
        close_conn = True
        raise e
    finally:
        db_pool.putconn(conn, close=close_conn)



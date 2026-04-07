CREATE SEQUENCE public.image_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;

CREATE SEQUENCE public.poses_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 9223372036854775807
	START 1
	CACHE 1
	NO CYCLE;

CREATE TABLE public.image (
	image_id int8 DEFAULT nextval('image_id_seq'::regclass) NOT NULL,
	"path" text NULL,
	CONSTRAINT image_pk PRIMARY KEY (image_id),
	CONSTRAINT path_unique UNIQUE (path)
);


CREATE TABLE public.poses (
	poses_id int8 DEFAULT nextval('poses_id_seq'::regclass) NOT NULL,
	image_id int8 NOT NULL,
	bbox_top_x float4 NULL,
	bbox_top_y float4 NULL,
	bbox_bottom_x float4 NULL,
	bbox_bottom_y float4 NULL,
	pose_vec public.vector NULL,
	person_num int4 NULL,
	norm_joint text NULL,
	CONSTRAINT image_person_unique UNIQUE (image_id, person_num),
	CONSTRAINT poses_pk PRIMARY KEY (poses_id)
);
CREATE INDEX pose_vec_hnsw_idx_1 ON public.poses USING hnsw (pose_vec vector_cosine_ops);
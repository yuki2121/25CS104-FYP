import { useEffect, useRef, useState } from "react";
import Masonry from "react-masonry-css";
import { skeletonToCoco18 } from "./pose/extract";
import { camera, skeleton } from "./PoseEditorCanvas";
import "./ResultDisplay.css";

const API_BASE = import.meta.env.VITE_API_BASE;
const PAGE_SIZE = 30;

type SearchPayload = {
	format: string;
	keypoints2d: any;
	score: any;
};

type ResultItem = {
	pose_id: string | number;
	url?: string;
	thumb_width?: number;
	thumb_height?: number;
};

const breakpointColumnsObj = {
	default: 6,
	1600: 5,
	1300: 4,
	1000: 3,
	700: 2,
	500: 1,
};

async function fetchSearchResults(
	payload: SearchPayload,
	limit: number,
	offset: number,
): Promise<ResultItem[]> {
	const res = await fetch("/api/search", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			...payload,
			limit,
			offset,
		}),
	});

	if (!res.ok) {
		const text = await res.text();
		throw new Error(`API ${res.status}: ${text}`);
	}

	const data = await res.json();
	return data.topK ?? [];
}

function buildSearchPayload(): SearchPayload | null {
	if (!skeleton || !camera) return null;

	const coco18pose = skeletonToCoco18(skeleton, camera);
	return {
		format: "openpose18",
		keypoints2d: coco18pose["xyz"],
		score: coco18pose["c"],
	};
}

function ResultDisplay() {
	const [results, setResults] = useState<ResultItem[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [loadingInitial, setLoadingInitial] = useState(false);
	const [loadingMore, setLoadingMore] = useState(false);
	const [hasMore, setHasMore] = useState(false);
	const [showGoTop, setShowGoTop] = useState(false);

	const sentinelRef = useRef<HTMLDivElement | null>(null);
	const payloadRef = useRef<SearchPayload | null>(null);
	const offsetRef = useRef(0);
	const loadingRef = useRef(false);

	const startNewSearch = async () => {
		const payload = buildSearchPayload();
		if (!payload) {
			setError("Skeleton or camera is not ready yet.");
			return;
		}

		payloadRef.current = payload;
		offsetRef.current = 0;
		loadingRef.current = true;

		setResults([]);
		setError(null);
		setHasMore(false);
		setLoadingInitial(true);

		try {
			const firstPage = await fetchSearchResults(payload, PAGE_SIZE, 0);
			setResults(firstPage);
			offsetRef.current = firstPage.length;
			setHasMore(firstPage.length === PAGE_SIZE);
		} catch (err: any) {
			setError(err.message ?? "Unknown error");
		} finally {
			loadingRef.current = false;
			setLoadingInitial(false);
		}
	};

	const loadMore = async () => {
		if (loadingRef.current || !hasMore || !payloadRef.current) return;

		loadingRef.current = true;
		setLoadingMore(true);

		try {
			const nextPage = await fetchSearchResults(
				payloadRef.current,
				PAGE_SIZE,
				offsetRef.current,
			);

			setResults((prev) => {
				const existing = new Set(prev.map((x) => x.pose_id));
				const deduped = nextPage.filter(
					(x) => !existing.has(x.pose_id),
				);
				return [...prev, ...deduped];
			});

			offsetRef.current += nextPage.length;
			setHasMore(nextPage.length === PAGE_SIZE);
		} catch (err: any) {
			setError(err.message ?? "Unknown error");
		} finally {
			loadingRef.current = false;
			setLoadingMore(false);
		}
	};

	useEffect(() => {
		const handleTriggerSearch = () => startNewSearch();
		window.addEventListener("trigger-pose-search", handleTriggerSearch);
		return () =>
			window.removeEventListener(
				"trigger-pose-search",
				handleTriggerSearch,
			);
	}, []);

	useEffect(() => {
		const sentinel = sentinelRef.current;
		if (!sentinel) return;

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries[0].isIntersecting) {
					loadMore();
				}
			},
			{
				root: document.getElementById("result-scroll-root"),
				rootMargin: "700px",
				threshold: 0,
			},
		);

		observer.observe(sentinel);
		return () => observer.disconnect();
	}, [hasMore]);

	useEffect(() => {
		const root = document.getElementById("result-scroll-root");
		if (!root) return;

		const handleScroll = () => {
			setShowGoTop(root.scrollTop > 300);
			console.log("Scroll position:", root.scrollTop);
		};

		root.addEventListener("scroll", handleScroll);
		handleScroll();

		return () => root.removeEventListener("scroll", handleScroll);
	}, []);

	const handleGoTop = () => {
		const root = document.getElementById("result-scroll-root");
		if (!root) return;
		root.scrollTo({ top: 0, behavior: "smooth" });
	};

	return (
		<div style={{ margin: 0, padding: 0 }}>
			{loadingInitial && <p>Loading...</p>}
			{error && <p style={{ color: "red" }}>Error: {error}</p>}

			{results.length > 0 && (
				<>
					<Masonry
						breakpointCols={breakpointColumnsObj}
						className="result-masonry-grid"
						columnClassName="result-masonry-grid_column"
					>
						{results.map((r) => {
							const aspectRatio =
								r.thumb_width && r.thumb_height
									? `${r.thumb_width} / ${r.thumb_height}`
									: undefined;

							return (
								<div key={r.pose_id} className="result-card">
									{r.url && (
										<img
											src={`${API_BASE}/dataset/thumbs/${r.pose_id}.jpg`}
											loading="eager"
											style={{
												width: "100%",
												display: "block",
												objectFit: "cover",
												background: "#eee",

												aspectRatio,
											}}
										/>
									)}

									<div className="result-card-id">
										ID: {r.pose_id}
									</div>
								</div>
							);
						})}
					</Masonry>

					<div ref={sentinelRef} className="result-footer">
						{loadingMore && "Loading more..."}
						{!loadingMore && !hasMore && "No more results"}
					</div>
				</>
			)}

			{showGoTop && (
				<button className="go-top-button" onClick={handleGoTop}>
					Top
				</button>
			)}
		</div>
	);
}

export default ResultDisplay;

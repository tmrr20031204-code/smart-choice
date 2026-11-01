const CACHE_NAME = 'smart-choice-shell-v2';
const PRECACHE_URLS = [
  '/',
  '/app_icon_v2.png',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(PRECACHE_URLS);
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            return caches.delete(name);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // POSTリクエスト、API呼び出し、外部オリジンは一切キャッシュせず素通し
  if (req.method !== 'GET' || url.pathname.startsWith('/api/') || url.origin !== self.location.origin) {
    return;
  }

  // ページナビゲーション（ルートアクセス等）のタイムアウト付きNetwork-First
  if (req.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(
      new Promise((resolve) => {
        let isResolved = false;

        // 1.5秒タイムアウトタイマー
        const timer = setTimeout(() => {
          if (!isResolved) {
            caches.match('/').then((cached) => {
              if (cached && !isResolved) {
                isResolved = true;
                resolve(cached);
              }
            });
          }
        }, 1500);

        // ネットワーク取得
        fetch(req).then((networkRes) => {
          clearTimeout(timer);
          if (networkRes && networkRes.status === 200) {
            const clone = networkRes.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put('/', clone));
          }
          if (!isResolved) {
            isResolved = true;
            resolve(networkRes);
          }
        }).catch(() => {
          clearTimeout(timer);
          if (!isResolved) {
            isResolved = true;
            caches.match('/').then((cached) => {
              if (cached) {
                resolve(cached);
              } else {
                resolve(new Response('Offline', { status: 503 }));
              }
            });
          }
        });
      })
    );
    return;
  }

  // 静的アセット（アイコン・マニフェスト）のキャッシュ優先 + バックグラウンド更新
  event.respondWith(
    caches.match(req).then((cached) => {
      const fetchPromise = fetch(req).then((networkRes) => {
        if (networkRes && networkRes.status === 200) {
          const clone = networkRes.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
        }
        return networkRes;
      }).catch(() => null);

      return cached || fetchPromise;
    })
  );
});

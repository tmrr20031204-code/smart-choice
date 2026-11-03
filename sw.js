const CACHE_NAME = 'smart-choice-shell-v3';
const PRECACHE_URLS = [
  '/',
  '/app_icon_v2.png',
  '/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return Promise.all(
        PRECACHE_URLS.map((url) =>
          fetch(new Request(url, { cache: 'reload' })).then((res) => {
            if (res && res.status === 200) {
              return cache.put(url, res);
            }
          })
        )
      );
    }).then(() => self.skipWaiting())
  );
});

function checkClientAlive(client) {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => resolve(false), 600);
    channel.port1.onmessage = (event) => {
      clearTimeout(timer);
      resolve(Boolean(event.data && event.data.alive));
    };
    try {
      client.postMessage({ type: 'CHECK_HEALTH' }, [channel.port2]);
    } catch (e) {
      clearTimeout(timer);
      resolve(false);
    }
  });
}

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const cacheNames = await caches.keys();
      const hadBrokenV2 = cacheNames.includes('smart-choice-shell-v2');
      await Promise.all(
        cacheNames.map((name) => {
          if (name !== CACHE_NAME) {
            return caches.delete(name);
          }
        })
      );
      await self.clients.claim();

      // 破損していたv2キャッシュ保持端末で、古いHTMLが表示されているタブのみ自動復旧
      if (hadBrokenV2) {
        const windowClients = await self.clients.matchAll({ type: 'window' });
        for (const client of windowClients) {
          const alive = await checkClientAlive(client);
          if (!alive && 'navigate' in client) {
            try {
              await client.navigate(client.url);
            } catch (e) {}
          }
        }
      }
    })()
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

        // 1.5秒タイムアウトタイマー（Renderスリープ時の体感0秒起動）
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

        // ネットワーク取得（HTTPキャッシュをバイパスして最新HTMLを取得）
        fetch(req, { cache: 'no-store' }).then((networkRes) => {
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

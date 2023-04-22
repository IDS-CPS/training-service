# IDS Training Service

## Starting Worker

```
watchfiles 'celery -A worker worker --loglevel=info -P solo' worker
```

## Starting Server

```
uvicorn main:app --reload
```
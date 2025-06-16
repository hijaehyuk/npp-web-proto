import urllib3
from server.auth.api import router as auth_router
from server.content.api import router as content_router
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.bbn_inference.api import router as bbn_router

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routing
app.include_router(auth_router)
app.include_router(content_router)
app.include_router(bbn_router)



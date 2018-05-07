git filter-branch --env-filter '
    if [ "$GIT_AUTHOR_NAME" != "Chris Schwartz" ]; then \
        export GIT_AUTHOR_NAME="Chris Schwartz" GIT_AUTHOR_EMAIL="chris.f.schwartz@gmail.com"; \
    fi
    '

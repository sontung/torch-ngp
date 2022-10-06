# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data MEDIUMBLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    return array.tobytes()


def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.full(4, np.NaN), prior_t=np.full(3, np.NaN), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, type(descriptors[0][0]))
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3),
                              qvec=np.array([1.0, 0.0, 0.0, 0.0]),
                              tvec=np.zeros(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
                                          array_to_blob(F), array_to_blob(E), array_to_blob(H),
                                          array_to_blob(qvec), array_to_blob(tvec)))


def example_usage():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    # Open the database.
    db = COLMAPDatabase.connect(args.database_path)

    # Read and check keypoints.
    keypoints = dict(
        (image_id, blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints"))
    id2name = dict((image_id, name)
                   for image_id, name in db.execute(
        "SELECT image_id, name FROM images"))
    desc = dict(
        (image_id, blob_to_array(data, np.uint8, (-1, 2)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM descriptors"))

    db.close()


def extract_id2name(database_path):
    db = COLMAPDatabase.connect(database_path)

    id2name = dict((image_id, name)
                   for image_id, name in db.execute(
        "SELECT image_id, name FROM images"))
    return id2name


def extract_colmap_sift(database_path):
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    # Read and check keypoints.
    try:
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 6)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM keypoints") if data is not None)
    except ValueError:
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 2)))
            for image_id, data in db.execute(
                "SELECT image_id, data FROM keypoints") if data is not None)

    desc = dict(
        (image_id, blob_to_array(data, np.uint8, (-1, 128)))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM descriptors") if data is not None)

    id2name = dict(
        (image_id, name)
        for image_id, name in db.execute(
            "SELECT image_id, name FROM images"))
    for im in keypoints:
        keypoints[im] = keypoints[im][:, :2]
    db.close()
    return keypoints, desc, id2name


def extract_colmap_hloc(database_path, kp_type, desc_type):
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    # Read and check keypoints.
    keypoints = dict(
        (image_id, blob_to_array(data, kp_type[0], (-1, kp_type[1])))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM keypoints") if data is not None)

    desc = dict(
        (image_id, np.transpose(blob_to_array(data, desc_type[0], (desc_type[1], -1))))
        for image_id, data in db.execute(
            "SELECT image_id, data FROM descriptors") if data is not None)

    id2name = dict(
        (image_id, name)
        for image_id, name in db.execute(
            "SELECT image_id, name FROM images"))
    for im in keypoints:
        keypoints[im] = keypoints[im][:, :2]
    db.close()
    return keypoints, desc, id2name


def extract_colmap_matches(database_path):
    """
        returns:
        pid2match: im1 im2 => fid1 fid2
    """
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    pid2rc = dict(
        (x, (y, z))
        for x, y, z in db.execute(
            "SELECT pair_id, rows, cols FROM matches"))
    pid2blob = dict(
        (x, y)
        for x, y in db.execute(
            "SELECT pair_id, data FROM matches"))
    pid2match = {}
    for pid in pid2rc:
        k = tuple(map(int, pair_id_to_image_ids(pid)))
        r, c = pid2rc[pid]
        blob = pid2blob[pid]
        if blob is None:
            pid2match[k] = None
        else:
            arr = blob_to_array(blob, np.uint32, (r, c))
            pid2match[k] = arr
    db.close()
    return pid2match


def extract_colmap_two_view_geometries(database_path):
    """
    returns:
    pid2match: im1 im2 => fid1 fid2
    pid2geom: im1 im2 => f mat, e mat, h mat
    """
    # Open the database.
    db = COLMAPDatabase.connect(database_path)

    pid2rc = dict(
        (x, (y, z))
        for x, y, z in db.execute(
            "SELECT pair_id, rows, cols FROM two_view_geometries"))
    pid2blob = dict(
        (x, y)
        for x, y in db.execute(
            "SELECT pair_id, data FROM two_view_geometries"))
    pid2match = {}
    for pid in pid2rc:
        k = tuple(map(int, pair_id_to_image_ids(pid)))
        r, c = pid2rc[pid]
        blob = pid2blob[pid]
        if blob is None:
            pid2match[k] = None
        else:
            arr = blob_to_array(blob, np.uint32, (r, c))
            pid2match[k] = arr

    pid2config = dict(
        (x, (y, a, b, c))
        for x, y, a, b, c in db.execute(
            "SELECT pair_id, config, F, E, H FROM two_view_geometries"))
    pid2geom = {}
    for pid in pid2config:
        k = tuple(map(int, pair_id_to_image_ids(pid)))
        config, f, e, h = pid2config[pid]
        arr_list = [None, None, None]
        if f is not None:
            arr1 = blob_to_array(f, np.float64, (3, 3))
            arr2 = blob_to_array(e, np.float64, (3, 3))
            arr3 = blob_to_array(h, np.float64, (3, 3))
            arr_list = [arr1, arr2, arr3]
        pid2geom[k] = (config, arr_list)
    db.close()
    return pid2match, pid2geom


if __name__ == "__main__":
    r2 = extract_colmap_matches("/home/n11373598/work/nerf-vloc/data/horn/images/colmap.db")
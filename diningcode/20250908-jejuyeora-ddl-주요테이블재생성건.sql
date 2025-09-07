-- 같은 세션에서만 적용됨
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS restaurant_category_mappings;
DROP TABLE IF EXISTS restaurant_menus;
DROP TABLE IF EXISTS restaurants;
DROP TABLE IF EXISTS regions;
DROP TABLE IF EXISTS restaurant_categories;

-- jejuyeora.regions definition

CREATE TABLE `regions` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '지역 식별 ID',
  `parent_id` int DEFAULT NULL COMMENT '상위 지역ID(시도 > 시군구 > 읍면동)',
  `name` json NOT NULL COMMENT '지역명({"ko":"제주도","en":"Jeju-do","cn":"濟州道"})',
  `level` int NOT NULL COMMENT '지역 레벨(1:시도, 2:시군구, 3:읍면동)',
  `code` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '지역코드',
  `center_latitude` decimal(10,8) DEFAULT NULL COMMENT '중심 위도',
  `center_longitude` decimal(11,8) DEFAULT NULL COMMENT '중심 경도',
  `place_count` int DEFAULT '0' COMMENT '지역 내 음식점 수',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '지역 생성 시간',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '지역 정보 수정시간',
  `deleted_at` datetime DEFAULT NULL COMMENT '지역 삭제 시간',
  `display_order` int DEFAULT '0' COMMENT '표시 순서',
  `is_active` tinyint(1) DEFAULT '0' COMMENT '노출 여부',
  PRIMARY KEY (`id`),
  UNIQUE KEY `UQ_code` (`code`),
  KEY `idx_level` (`level`),
  KEY `idx_code` (`code`),
  KEY `idx_parent` (`parent_id`),
  CONSTRAINT `FK_region_TO_region` FOREIGN KEY (`parent_id`) REFERENCES `regions` (`id`) ON DELETE SET NULL,
  CONSTRAINT `regions_chk_1` CHECK ((`level` in (1,2,3)))
) ENGINE=InnoDB AUTO_INCREMENT=69 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='지역';

-- jejuyeora.restaurant_categories definition

CREATE TABLE `restaurant_categories` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '카테고리 ID',
  `parent_id` int DEFAULT NULL COMMENT '상위 카테고리 ID',
  `name` json NOT NULL COMMENT '카테고리명({"ko":"한식","en":"Korean Food","cn":"韓國食品"})',
  `icon_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '아이콘 URL',
  `display_order` int DEFAULT '0' COMMENT '표시 순서',
  `is_active` tinyint(1) DEFAULT '0' COMMENT '노출 여부',
  `is_default` tinyint(1) DEFAULT '0' COMMENT '기본 카테고리 여부',
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '카테고리 설명',
  `item_count` int DEFAULT '0' COMMENT '카테고리 아이템 수',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '카테고리 생성 시간',
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '카테고리 수정 시간',
  `deleted_at` datetime DEFAULT NULL COMMENT '카테고리 삭제 시간',
  PRIMARY KEY (`id`),
  KEY `idx_parent` (`parent_id`),
  KEY `idx_active_order` (`is_active`,`display_order`),
  CONSTRAINT `FK_rcats_TO_rcats` FOREIGN KEY (`parent_id`) REFERENCES `restaurant_categories` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB AUTO_INCREMENT=1018 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='음식점 카테고리';

-- jejuyeora.restaurants definition

CREATE TABLE `restaurants` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '음식점 식별 ID',
  `name` json NOT NULL COMMENT '음식점명({"ko":"...","en":"...","cn":"..."})',
  `description` json DEFAULT NULL COMMENT '상세 소개',
  `short_description` json DEFAULT NULL COMMENT '한줄 소개',
  `phone` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '전화번호',
  `address` json NOT NULL COMMENT '전체 주소',
  `address_detail` json DEFAULT NULL COMMENT '상세 주소',
  `latitude` decimal(10,8) DEFAULT NULL COMMENT '위도',
  `longitude` decimal(11,8) DEFAULT NULL COMMENT '경도',
  `operating_times` json DEFAULT NULL COMMENT '운영시간',
  `business_hours` json DEFAULT NULL COMMENT '영업시간',
  `last_order_time` json DEFAULT NULL COMMENT '라스트 오더',
  `break_time` json DEFAULT NULL COMMENT '브레이크 타임',
  `reservation_hours` json DEFAULT NULL COMMENT '예약 가능 시간',
  `waiting_enabled` tinyint(1) DEFAULT NULL COMMENT '줄서기 활성화',
  `reservation_enabled` tinyint(1) DEFAULT NULL COMMENT '예약 활성화',
  `reservation_deposit_required` tinyint(1) DEFAULT NULL COMMENT '예약 보증금 필요',
  `reservation_deposit_amount` int DEFAULT NULL COMMENT '예약 보증 금액',
  `status` enum('PENDING','ACTIVE','INACTIVE','CLOSED') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL DEFAULT 'PENDING' COMMENT '조회 상태',
  `main_image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT 'default_image_url' COMMENT '대표 이미지 URL',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '생성',
  `updated_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정',
  `deleted_at` datetime DEFAULT NULL COMMENT '삭제',
  `owner_id` bigint NOT NULL COMMENT '사장님 userID',
  `region_id` int NOT NULL COMMENT '지역 식별 ID',
  `avg_wait_time_per_party` int NOT NULL DEFAULT '3' COMMENT '팀당 웨이팅 시간',
  `min_party_size` int NOT NULL DEFAULT '1' COMMENT '최소 예약 인원',
  `max_party_size` int NOT NULL DEFAULT '8' COMMENT '최대 예약 인원',
  `min_adult_count` int DEFAULT NULL COMMENT '최소 성인 예약 인원',
  `reservation_interval` int DEFAULT '30' COMMENT '예약 슬롯 간격',
  `noti_party_left` int DEFAULT '0',
  `business_representative_name` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '사업자 대표자명',
  `business_name` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '상호명',
  `business_address` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '사업자 주소',
  `business_registration_number` varchar(20) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '사업자등록번호',
  PRIMARY KEY (`id`),
  KEY `idx_status` (`status`),
  KEY `idx_location` (`latitude`,`longitude`),
  KEY `idx_owner` (`owner_id`),
  KEY `idx_region` (`region_id`),
  CONSTRAINT `FK_region_TO_restaurants` FOREIGN KEY (`region_id`) REFERENCES `regions` (`id`),
  CONSTRAINT `FK_users_TO_restaurants` FOREIGN KEY (`owner_id`) REFERENCES `users` (`id`),
  CONSTRAINT `restaurants_chk_1` CHECK (((`latitude` is null) or (`latitude` between -(90.0) and 90.0))),
  CONSTRAINT `restaurants_chk_2` CHECK (((`longitude` is null) or (`longitude` between -(180.0) and 180.0)))
) ENGINE=InnoDB AUTO_INCREMENT=2683 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='음식점 정보';

-- jejuyeora.restaurant_menus definition

CREATE TABLE `restaurant_menus` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '메뉴 ID',
  `restaurant_id` bigint NOT NULL COMMENT '음식점 식별 ID',
  `name` json NOT NULL COMMENT '메뉴명({"ko":"...","en":"...","cn":"..."})',
  `description` json DEFAULT NULL COMMENT '메뉴 설명',
  `price` int NOT NULL COMMENT '가격',
  `discount_price` int DEFAULT NULL COMMENT '할인 가격',
  `image_url` varchar(500) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '메뉴 이미지 URL',
  `is_signature` tinyint(1) DEFAULT '0' COMMENT '대표 메뉴 여부',
  `is_popular` tinyint(1) DEFAULT '0' COMMENT '인기 메뉴 여부',
  `is_new` tinyint(1) DEFAULT '0' COMMENT '신메뉴 여부',
  `status` enum('ACTIVE','HIDDEN','SOLD_OUT','INACTIVE') CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT 'ACTIVE' COMMENT '상태',
  `allergen_info` json DEFAULT NULL COMMENT '알레르기 정보',
  `display_order` int DEFAULT '0' COMMENT '표시 순서',
  `order_count` int DEFAULT '0' COMMENT '주문 횟수',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '생성',
  `updated_at` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정',
  `deleted_at` datetime DEFAULT NULL COMMENT '삭제',
  PRIMARY KEY (`id`),
  KEY `idx_place_category` (`restaurant_id`),
  KEY `idx_place_order` (`restaurant_id`,`display_order`),
  KEY `idx_popular` (`restaurant_id`,`is_popular`,`status`),
  CONSTRAINT `FK_restaurants_TO_restaurant_menus` FOREIGN KEY (`restaurant_id`) REFERENCES `restaurants` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=31789 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='음식점 메뉴';

-- jejuyeora.restaurant_category_mappings definition

CREATE TABLE `restaurant_category_mappings` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `restaurant_id` bigint NOT NULL COMMENT '음식점 식별 ID',
  `category_id` int NOT NULL COMMENT '카테고리 ID',
  `is_primary` tinyint(1) DEFAULT '0' COMMENT '대표 카테고리 여부',
  `created_at` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '생성 시간',
  PRIMARY KEY (`id`),
  UNIQUE KEY `restaurant_category_mappings_unique` (`restaurant_id`,`category_id`),
  KEY `idx_cat` (`category_id`),
  CONSTRAINT `FK_rcats_TO_rcmaps` FOREIGN KEY (`category_id`) REFERENCES `restaurant_categories` (`id`) ON DELETE CASCADE,
  CONSTRAINT `FK_restaurants_TO_rcmaps` FOREIGN KEY (`restaurant_id`) REFERENCES `restaurants` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=5076 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='음식점-카테고리 매핑';

SET FOREIGN_KEY_CHECKS = 1;

-- 필요시 시스템 계정 추가건 

INSERT INTO users (
    username,
    password,
    social_type,
    social_id,
    email,
    nickname,
    phone_number,
    profile_image_url,
    status,
    language,
    notification_settings,
    required_terms_agreement,
    role,
    created_at,
    updated_at
) VALUES (
    'system',                                    -- username
    'system2580!-sha256',                                  -- password (실제는 bcrypt 등 해시 필요)
    NULL,                                            -- social_type
    NULL,                                            -- social_id
    'testuser01@example.com',                        -- email
    '시스템유저',                                      -- nickname
    '01040425279',                                   -- phone_number
    'https://s3/default_profile_img.png',            -- profile_image_url
    'ACTIVE',                                        -- status
    'ko',                                            -- language
    JSON_OBJECT(
        'marketing', 'Y',
        'reservation', 'Y',
        'waiting', 'Y',
        'review_reply', 'Y',
        'event', 'Y',
        'night', 'N'
    ),                                               -- notification_settings
    1,                                               -- required_terms_agreement
    'USER',                                          -- role
    NOW(),                                           -- created_at
    NOW()                                            -- updated_at
);
